import gradio as gr
import subprocess
import os
import queue


class TerminalSession:
    def __init__(self, cwd=None):
        self.cwd = cwd or os.getcwd()
        self.history = []
        self.command_queue = queue.Queue()

    def execute_command(self, command):
        """Execute a shell command and return the output"""
        if not command.strip():
            return ""

        # Handle 'cd' command separately since it changes directory
        if command.strip().startswith('cd '):
            path = command.strip()[3:].strip()
            try:
                if path == '~':
                    new_path = os.path.expanduser('~')
                elif path.startswith('~'):
                    new_path = os.path.expanduser(path)
                elif os.path.isabs(path):
                    new_path = path
                else:
                    new_path = os.path.join(self.cwd, path)

                new_path = os.path.normpath(new_path)

                if os.path.isdir(new_path):
                    self.cwd = new_path
                    return f"Changed directory to: {self.cwd}"
                else:
                    return f"cd: no such file or directory: {path}"
            except Exception as e:
                return f"cd: {str(e)}"

        # Handle 'clear' command
        if command.strip() == 'clear':
            return "CLEAR_TERMINAL"

        # Handle 'pwd' command
        if command.strip() == 'pwd':
            return self.cwd

        # Execute other commands
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                executable='/bin/zsh'
            )

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += result.stderr

            return output.strip() if output.strip() else "(Command executed successfully)"

        except subprocess.TimeoutExpired:
            return "Error: Command timed out (30s limit)"
        except Exception as e:
            return f"Error: {str(e)}"


# Global terminal session
terminal_session = TerminalSession()


def format_terminal_output(history):
    """Format the terminal history for display"""
    output = []
    for cmd, result in history:
        output.append(f"$ {cmd}")
        if result:
            output.append(result)
        output.append("")  # Empty line for spacing

    return "\n".join(output)


def process_command(command, current_output):
    """Process a command and update the terminal output"""
    if not command.strip():
        return current_output, ""

    # Execute the command
    result = terminal_session.execute_command(command)

    # Handle clear command
    if result == "CLEAR_TERMINAL":
        terminal_session.history = []
        return "", ""

    # Add to history
    terminal_session.history.append((command, result))

    # Format output
    output = format_terminal_output(terminal_session.history)

    return output, ""


def tab_terminal():
    """Create the terminal tab interface"""
    gr.Markdown("""
    ## Terminal

    Execute shell commands directly from the web interface.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            terminal_output = gr.Textbox(
                label="",
                lines=20,
                max_lines=30,
                interactive=False,
                show_label=False,
                placeholder="Terminal output will appear here...\nType a command below and press Enter or click Run.",
                elem_id="terminal-output",
                container=False
            )

    with gr.Row():
        with gr.Column(scale=9):
            command_input = gr.Textbox(
                label="",
                placeholder="Enter command here",
                show_label=False,
                max_lines=1,
                container=False
            )
        with gr.Column(scale=1, min_width=100, elem_id="run-button-column"):
            run_button = gr.Button("Run", variant="primary", size="sm")

    # Event handlers
    def run_command_handler(cmd, output):
        return process_command(cmd, output)

    # Connect events
    run_button.click(
        fn=run_command_handler,
        inputs=[command_input, terminal_output],
        outputs=[terminal_output, command_input]
    )

    command_input.submit(
        fn=run_command_handler,
        inputs=[command_input, terminal_output],
        outputs=[terminal_output, command_input]
    )
