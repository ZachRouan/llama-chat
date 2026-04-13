# main.py
"""Application entry point — main loop, command dispatch, signal handling."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import termios
import time
from dataclasses import dataclass, field

import httpx

from chat import ChatSession
from client import LlamaClient, ServerStatus, ModelInfo
from config import (
    AGENT_SYSTEM_PROMPT,
    AppConfig,
    ServerConfig,
    load_config,
    run_first_time_setup,
    ENV_PATH,
    HISTORY_DIR,
    SESSION_PATH,
)
import ui
from pathlib import Path

from tools import TOOL_DEFINITIONS, clean_arguments, execute_tool, execute_command
from permissions import load_permissions, check_permission, add_allow_rule


@dataclass
class ChatState:
    """Mutable state passed through the chat loop."""

    client: LlamaClient
    session: ChatSession
    server: str  # "host:port"
    model: str  # model id
    context_length: int
    config: AppConfig
    agent_mode: bool = False


# ---------------------------------------------------------------------------
# Context length resolution
# ---------------------------------------------------------------------------

async def _resolve_context_length(
    client: LlamaClient,
    config: AppConfig,
    server_config: ServerConfig,
) -> int:
    """Resolve context length: per-server override > /props > fallback."""
    if server_config.context_length is not None:
        return server_config.context_length
    auto = await client.get_context_length()
    if auto is not None:
        return auto
    return config.context_length


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

async def select_model(
    config: AppConfig,
) -> tuple[LlamaClient, str, str, int, ServerConfig] | None:
    """Query all servers, show menu, handle user choice.

    Returns (client, server_str, model_id, context_length, server_config) or
    None if all servers are offline.
    """
    menu_items: list[dict] = []
    # Parallel: (server_config, status, models)
    server_data: list[tuple[ServerConfig, ServerStatus, list[ModelInfo]]] = []

    clients: list[LlamaClient] = []
    try:
        for sc in config.servers:
            clients.append(LlamaClient(sc.host, sc.port))

        # Check health in parallel
        statuses = await asyncio.gather(*(c.check_health() for c in clients))

        # Get models for online servers in parallel
        online_indices = [
            i for i, s in enumerate(statuses) if s == ServerStatus.ONLINE
        ]
        if online_indices:
            model_results = await asyncio.gather(
                *(clients[i].get_models() for i in online_indices),
                return_exceptions=True,
            )
            models_by_index: dict[int, list[ModelInfo]] = {}
            for i, result in zip(online_indices, model_results):
                models_by_index[i] = result if not isinstance(result, Exception) else []
        else:
            models_by_index = {}

        for i, (sc, status) in enumerate(zip(config.servers, statuses)):
            server_data.append((sc, status, models_by_index.get(i, [])))

    except Exception:
        # Clean up clients on unexpected errors
        for c in clients:
            await c.close()
        raise

    # Build menu
    selectable: set[int] = set()
    number = 0
    for sc, status, models in server_data:
        number += 1
        server_str = f"{sc.host}:{sc.port}"
        if status == ServerStatus.ONLINE and models:
            label = models[0].id
            menu_items.append({
                "number": number,
                "label": label,
                "server": server_str,
                "status": "online",
            })
            selectable.add(number)
        elif status == ServerStatus.LOADING:
            menu_items.append({
                "number": number,
                "label": server_str,
                "server": server_str,
                "status": "loading",
            })
            selectable.add(number)
        else:
            menu_items.append({
                "number": number,
                "label": server_str,
                "server": server_str,
                "status": "offline",
            })

    # All offline?
    if not selectable:
        offline_list = ", ".join(
            f"{sc.host}:{sc.port}" for sc, status, _ in server_data
        )
        ui.print_error(f"All servers are offline: {offline_list}")
        for c in clients:
            await c.close()
        return None

    ui.print_model_menu(menu_items)
    choice = ui.get_model_choice(len(menu_items), selectable)
    idx = choice - 1
    sc, status, models = server_data[idx]
    chosen_client = clients[idx]

    # Close unchosen clients
    for i, c in enumerate(clients):
        if i != idx:
            await c.close()

    # If loading, wait for it
    if status == ServerStatus.LOADING:
        ui.print_muted(f"Waiting for {sc.host}:{sc.port} to finish loading...")
        try:
            while True:
                await asyncio.sleep(2)
                new_status = await chosen_client.check_health()
                if new_status == ServerStatus.ONLINE:
                    break
                if new_status == ServerStatus.OFFLINE:
                    ui.print_error(f"Server {sc.host}:{sc.port} went offline.")
                    await chosen_client.close()
                    return None
        except (KeyboardInterrupt, EOFError, asyncio.CancelledError):
            await chosen_client.close()
            # Return to menu by recursing
            return await select_model(config)

    # Get models if we didn't have them (loading case)
    if not models:
        models = await chosen_client.get_models()
        if not models:
            ui.print_error(f"No models found on {sc.host}:{sc.port}.")
            await chosen_client.close()
            return None

    model_id = models[0].id
    server_str = f"{sc.host}:{sc.port}"
    context_length = await _resolve_context_length(chosen_client, config, sc)

    return chosen_client, server_str, model_id, context_length, sc


# ---------------------------------------------------------------------------
# Session save helper
# ---------------------------------------------------------------------------

def _save_session(state: ChatState) -> None:
    """Save session if it has messages."""
    if not state.session.is_empty():
        state.session.save(SESSION_PATH, state.server, state.model)


# ---------------------------------------------------------------------------
# Queue detection background task
# ---------------------------------------------------------------------------

async def _queue_updater(
    spinner: ui.SpinnerDisplay,
    server: str,
    started: float,
) -> None:
    """After 3s of no cancellation, flip the spinner to 'Queued' with elapsed time."""
    await asyncio.sleep(3)
    while True:
        elapsed = time.monotonic() - started
        spinner.set_queued(server, elapsed)
        await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# Terminal state
# ---------------------------------------------------------------------------

_saved_terminal_attrs: list | None = None


def _save_terminal() -> None:
    """Save terminal attributes so they can be restored on abrupt exit."""
    global _saved_terminal_attrs
    try:
        _saved_terminal_attrs = termios.tcgetattr(sys.stdin.fileno())
    except (termios.error, ValueError, OSError):
        pass


def _restore_terminal() -> None:
    """Restore terminal attributes (echo, canonical mode, etc.)."""
    if _saved_terminal_attrs is not None:
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _saved_terminal_attrs)
        except (termios.error, ValueError, OSError):
            pass


# ---------------------------------------------------------------------------
# Send message
# ---------------------------------------------------------------------------

MAX_TOOL_ITERATIONS = 15


def _build_api_messages(state: ChatState) -> list[dict]:
    """Build API messages, prepending agent system prompt if agent mode is on."""
    messages = state.session.get_messages_for_api()
    if state.agent_mode:
        messages = list(messages)
        messages[0] = {
            "role": "system",
            "content": AGENT_SYSTEM_PROMPT + "\n\n" + messages[0]["content"],
        }
    return messages


async def send_message(state: ChatState, user_input: str) -> str | None:
    """Send a message and stream the response.

    When agent_mode is on, runs a tool loop: the model can call tools
    and the results are sent back until it responds with plain text.

    Returns None on success, or a prefill string if cancelled/failed.
    """
    message_count_before = state.session.message_count
    state.session.add_message("user", user_input)

    try:
        for iteration in range(MAX_TOOL_ITERATIONS):
            spinner = ui.SpinnerDisplay()
            spinner.start()
            queue_task: asyncio.Task | None = None
            stream_display = ui.StreamingDisplay()
            stream = None

            try:
                started = time.monotonic()
                queue_task = asyncio.create_task(
                    _queue_updater(spinner, state.server, started)
                )

                # Build messages with agent system prompt if needed
                messages = _build_api_messages(state)

                # Count tokens and truncate if needed
                prompt_tokens = await state.client.count_messages_tokens(messages)
                pairs_dropped = state.session.truncate_if_needed(prompt_tokens)

                if pairs_dropped > 0:
                    messages = _build_api_messages(state)
                    prompt_tokens = await state.client.count_messages_tokens(messages)

                stream = await state.client.stream_chat(
                    messages=messages,
                    temperature=state.config.temperature,
                    max_tokens=state.config.max_tokens,
                    tools=TOOL_DEFINITIONS if state.agent_mode else None,
                )
                stream.set_prompt_tokens(prompt_tokens)

                first_token = True
                async for token, is_reasoning in stream:
                    if first_token:
                        queue_task.cancel()
                        spinner.stop()
                        await asyncio.sleep(0)  # let cancellation propagate
                        stream_display.start()
                        first_token = False
                    stream_display.update(token, is_reasoning)

                # Stream finished — ensure queue task is dead
                if first_token:
                    queue_task.cancel()
                    spinner.stop()
                else:
                    stream_display.stop()
                # Wait for queue task cleanup to avoid stale Live state
                try:
                    await queue_task
                except (asyncio.CancelledError, Exception):
                    pass

                # Handle empty/interrupted
                if stream.is_empty and not stream.has_tool_calls:
                    state.session.remove_last_message()
                    ui.print_muted("(empty response)")
                    return user_input

                if stream.was_interrupted:
                    stream_display.stop()
                    state.session.remove_last_message()
                    ui.print_muted("(stream interrupted)")
                    return user_input

                # Check for tool calls
                if not stream.has_tool_calls:
                    # Normal text response
                    state.session.add_message("assistant", stream.content)
                    ui.print_stats(
                        stream.final_token_count,
                        stream.duration,
                        stream.hit_max_tokens,
                        stream.total_context_tokens,
                        state.context_length,
                    )
                    return None

                # Tool calls — add assistant message and execute
                state.session.add_message(
                    "assistant",
                    content=stream.content if stream.content else None,
                    tool_calls=stream.tool_calls,
                )

                for tool_call in stream.tool_calls:
                    name = tool_call["function"]["name"]
                    raw_arguments = tool_call["function"]["arguments"]
                    tool_call_id = tool_call["id"]

                    try:
                        arguments = clean_arguments(raw_arguments)
                    except Exception:
                        arguments = {}
                        result = f"Error: Failed to parse tool arguments: {raw_arguments}"
                        ui.print_tool_call(name, arguments)
                        ui.print_tool_result(name, result)
                        state.session.add_message("tool", result, tool_call_id=tool_call_id, name=name)
                        continue

                    ui.print_tool_call(name, arguments)

                    # Permission check
                    working_directory = Path.cwd()
                    permissions = load_permissions(working_directory)
                    permission = check_permission(permissions, name, arguments)
                    if permission == "ask":
                        # Build summary for prompt
                        if name == "write_file":
                            path = arguments.get("path", "")
                            content = arguments.get("content", "")
                            line_count = content.count("\n") + 1
                            summary = f"{path} ({line_count} lines)"
                        elif name == "run_command":
                            summary = arguments.get("command", "")
                        else:
                            summary = str(arguments)[:100]

                        response = await asyncio.to_thread(
                            ui.prompt_tool_permission, name, summary
                        )
                        if response == "no":
                            result = "User denied this tool call."
                            ui.print_tool_result(name, result)
                            state.session.add_message("tool", result, tool_call_id=tool_call_id, name=name)
                            continue
                        elif response == "always":
                            if name == "write_file":
                                derived_parent = str(Path(arguments.get("path", "")).parent)
                                if derived_parent == ".":
                                    ui.print_broad_pattern_warning(name)
                            add_allow_rule(working_directory, name, arguments)

                    if name == "run_command":
                        # Stream command output
                        command = arguments.get("command", "")
                        output_lines: list[str] = []
                        async for line in execute_command(command):
                            output_lines.append(line)
                            ui.print_command_output(line)
                        result = "".join(output_lines)
                    else:
                        result = execute_tool(name, arguments)

                    ui.print_tool_result(name, result)
                    state.session.add_message("tool", result, tool_call_id=tool_call_id, name=name)

                # Loop continues — next iteration will send updated history

            except asyncio.CancelledError:
                if queue_task:
                    queue_task.cancel()
                spinner.stop()
                stream_display.stop()
                state.session.rollback_to(message_count_before)
                return user_input

            except (httpx.ConnectError, httpx.ConnectTimeout) as error:
                if queue_task:
                    queue_task.cancel()
                spinner.stop()
                stream_display.stop()
                state.session.rollback_to(message_count_before)
                ui.print_error(f"Connection failed: {error}")
                return user_input

            except httpx.HTTPStatusError as error:
                if queue_task:
                    queue_task.cancel()
                spinner.stop()
                stream_display.stop()
                state.session.rollback_to(message_count_before)
                ui.print_error(f"Server error: {error.response.status_code}")
                return user_input

        else:
            # Hit iteration cap
            ui.print_error(
                f"Agent hit the {MAX_TOOL_ITERATIONS}-iteration limit. "
                "Showing last response — you can tell it to continue."
            )
            if stream and stream.content:
                ui.console.print()
            return None

    except asyncio.CancelledError:
        # Outer cancellation (e.g., during tool execution)
        state.session.rollback_to(message_count_before)
        return user_input


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------

async def handle_command(cmd: str, args: str, state: ChatState) -> bool:
    """Handle a / command. Returns False for /quit, True otherwise."""
    if cmd == "/quit":
        _save_session(state)
        ui.print_muted("Session saved. Goodbye!")
        return False

    elif cmd == "/help":
        ui.print_help(str(ENV_PATH))
        return True

    elif cmd == "/clear":
        if state.session.is_empty():
            ui.print_muted("Nothing to clear.")
        else:
            _save_session(state)
            ChatSession.archive(SESSION_PATH, HISTORY_DIR)
            state.session.clear()
            ui.print_muted("Conversation cleared.")
        return True

    elif cmd == "/system":
        if args:
            state.session.set_system_prompt(args)
            ui.print_muted("System prompt updated.")
        else:
            ui.print_muted(f"Current system prompt: {state.session.system_prompt}")
        return True

    elif cmd == "/model":
        result = await select_model(state.config)
        if result is None:
            return True
        new_client, new_server, new_model, new_ctx, _ = result

        # Warn if context shrinks significantly
        estimated = state.session.estimate_tokens()
        if estimated > new_ctx * 0.5:
            msg = (
                f"Current conversation (~{estimated} tokens) may not fit in "
                f"{new_model}'s context ({new_ctx:,} tokens). "
                f"Oldest messages will be truncated. Continue?"
            )
            if not ui.confirm(msg):
                await new_client.close()
                return True

        # Switch
        await state.client.close()
        state.client = new_client
        state.server = new_server
        state.model = new_model
        state.context_length = new_ctx
        state.session.set_context_length(new_ctx)
        ui.print_muted(f"Switched to {new_model} on {new_server}.")
        return True

    elif cmd == "/agent":
        if args.lower() == "on":
            state.agent_mode = True
        elif args.lower() == "off":
            state.agent_mode = False
        else:
            state.agent_mode = not state.agent_mode
        status = "on" if state.agent_mode else "off"
        ui.print_muted(f"Agent mode {status}.")
        return True

    else:
        ui.print_muted("Unknown command. Type /help for available commands.")
        return True


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

async def chat_loop(state: ChatState) -> None:
    """Main chat loop with signal handling."""
    _save_terminal()
    loop = asyncio.get_running_loop()
    generation_task: asyncio.Task | None = None
    last_sigint: float = 0.0
    prefill: str = ""

    def sigint_handler() -> None:
        nonlocal last_sigint, generation_task
        now = time.monotonic()

        if generation_task is not None and not generation_task.done():
            generation_task.cancel()
            return

        # At prompt: double-press within 3 seconds
        if now - last_sigint < 3.0:
            _save_session(state)
            _restore_terminal()
            sys.stdout.write("\nSession saved. Goodbye!\n")
            sys.stdout.flush()
            os._exit(0)

        last_sigint = now
        sys.stdout.write("\nPress Ctrl+C again to quit.\n")
        sys.stdout.flush()

    loop.add_signal_handler(signal.SIGINT, sigint_handler)

    try:
        while True:
            try:
                prompt = "Agent > " if state.agent_mode else "You > "
                user_input = await asyncio.to_thread(ui.get_input, prompt, prefill)
                prefill = ""
            except (EOFError, KeyboardInterrupt):
                # EOFError = Ctrl+D, KeyboardInterrupt = signal leaked to thread
                continue

            user_input = user_input.strip()
            if not user_input:
                continue

            # Command dispatch
            if user_input.startswith("/"):
                parts = user_input.split(None, 1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                keep_going = await handle_command(cmd, args, state)
                if not keep_going:
                    break
                continue

            # Send message as a task so signal handler can cancel it
            generation_task = asyncio.create_task(send_message(state, user_input))
            result = await generation_task
            generation_task = None

            if result is not None:
                prefill = result
    finally:
        loop.remove_signal_handler(signal.SIGINT)


# ---------------------------------------------------------------------------
# Resume flow
# ---------------------------------------------------------------------------

async def _try_resume(config: AppConfig) -> tuple[ChatState | None, ChatSession | None]:
    """Attempt to resume a previous session.

    Returns:
        (ChatState, None) on full resume,
        (None, ChatSession) to keep history but switch model,
        or (None, None) to start fresh.
    """
    if not SESSION_PATH.exists():
        return None, None

    session, meta = ChatSession.load(SESSION_PATH, config.max_tokens)
    saved_server = meta["server"]
    saved_model = meta["model"]

    msg = f"Resume previous session with {saved_model} on {saved_server}?"
    if not ui.confirm(msg):
        ChatSession.archive(SESSION_PATH, HISTORY_DIR)
        return None, None

    # Parse saved server
    parts = saved_server.split(":")
    host, port = parts[0], int(parts[1])

    # Find matching server config for context length override
    server_config = None
    for sc in config.servers:
        if sc.host == host and sc.port == port:
            server_config = sc
            break

    client = LlamaClient(host, port)
    status = await client.check_health()

    if status == ServerStatus.OFFLINE:
        await client.close()
        ui.print_muted(f"Server {saved_server} is offline. Selecting a new model...")
        return None, session  # keep history

    # Server is online or loading — check model
    if status == ServerStatus.LOADING:
        ui.print_muted(f"Waiting for {saved_server} to finish loading...")
        while True:
            await asyncio.sleep(2)
            status = await client.check_health()
            if status == ServerStatus.ONLINE:
                break
            if status == ServerStatus.OFFLINE:
                await client.close()
                ui.print_muted(f"Server {saved_server} went offline. Selecting a new model...")
                return None, session  # keep history

    models = await client.get_models()
    current_model = models[0].id if models else None

    if current_model != saved_model:
        msg = (
            f"Server is now running {current_model} "
            f"(session was with {saved_model}). Continue anyway?"
        )
        if not ui.confirm(msg):
            await client.close()
            return None, session  # keep history, switch model

    # Resolve context length
    if server_config:
        ctx = await _resolve_context_length(client, config, server_config)
    else:
        auto = await client.get_context_length()
        ctx = auto if auto is not None else config.context_length

    session.set_context_length(ctx)
    model_id = current_model or saved_model
    return ChatState(
        client=client,
        session=session,
        server=saved_server,
        model=model_id,
        context_length=ctx,
        config=config,
    ), None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    """Application entry point."""
    # Load config
    try:
        if ENV_PATH.exists():
            config = load_config()
        else:
            config = run_first_time_setup()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return

    # Resume check
    state: ChatState | None = None
    reuse_session: ChatSession | None = None
    try:
        state, reuse_session = await _try_resume(config)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return

    # Model selection if no resumed state
    if state is None:
        try:
            result = await select_model(config)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return

        if result is None:
            return

        client, server, model_id, context_length, server_config = result

        if reuse_session is not None:
            # Keep conversation history from declined resume
            session = reuse_session
            session.set_context_length(context_length)
        else:
            session = ChatSession(
                system_prompt=config.system_prompt,
                context_length=context_length,
                max_tokens=config.max_tokens,
            )

        state = ChatState(
            client=client,
            session=session,
            server=server,
            model=model_id,
            context_length=context_length,
            config=config,
        )

    # Banner
    ui.print_banner(state.model, state.server, state.context_length)

    # Chat loop
    try:
        await chat_loop(state)
    finally:
        await state.client.close()


if __name__ == "__main__":
    asyncio.run(main())
