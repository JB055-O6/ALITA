from pathlib import Path
import typer
from rich.console import Console
from rich.progress import Progress
import subprocess
import sys

from .config import Config
from .core.brain import Brain
from .core.voice import VoiceInterface
from .core.vision import VisionSystem
from .core.automation import SystemControl
from .utils.safety import SafetyManager

app = typer.Typer()
console = Console()

@app.command()
def start(
    config_path: Path = typer.Option(None, "--config", "-c", help="Path to config file"),
    dev: bool = typer.Option(False, "--dev", help="Enable development mode"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
):
    """Start the Alita AI Assistant."""
    config = Config.load(config_path)
    if dev:
        config.system.debug = True
    
    with console.status("Initializing AI systems..."):
        brain = Brain(config.ai)
        voice = VoiceInterface(config.voice)
        vision = VisionSystem(config.vision)
        system = SystemControl(config.system)
        safety = SafetyManager()
    
    console.print("🚀 [bold green]Alita is ready![/]")
    console.print(f"Wake word: '{config.voice.wake_word}'")
    console.print("Kill switch: Alt+Shift+K")
    
    try:
        # Start main event loop
        voice.start_listening()
        while True:
            command = voice.get_command()
            if command:
                # Process through brain
                plan = brain.plan(command)
                if safety.check_plan(plan):
                    system.execute(plan)
    except KeyboardInterrupt:
        console.print("\n👋 Shutting down safely...")
    finally:
        voice.stop()
        system.cleanup()

@app.command()
def download(
    what: str = typer.Argument(..., help="What to download: 'models' or 'voices'"),
):
    """Download required models and voices."""
    if what == "models":
        with Progress() as progress:
            # Download core models
            task1 = progress.add_task("Downloading LLaMA...", total=100)
            # TODO: Implement actual download
            progress.update(task1, completed=100)
            
            task2 = progress.add_task("Downloading Whisper...", total=100)
            # TODO: Implement actual download
            progress.update(task2, completed=100)
    elif what == "voices":
        subprocess.run([sys.executable, "-m", "TTS", "train", "--download_dir", "models/voices"])
    else:
        console.print(f"[red]Unknown download type: {what}")

def main():
    """Entry point for the CLI."""
    app()