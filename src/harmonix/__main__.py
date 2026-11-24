"""Package entry point for the Pygame shell milestone."""

from .ui.pygame_app import PygameApp


def main() -> None:
    """Start the minimal Pygame application shell."""
    PygameApp().run()


if __name__ == "__main__":
    main()