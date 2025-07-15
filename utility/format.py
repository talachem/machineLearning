class Format():
    """
    This class provides various formatting options for console output.
    """
    def __init__(self) -> None:
        self.colors = {
            'black': '\033[30m',
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'purple': '\033[35m',
            'cyan': '\033[36m',
            'white': '\033[37m',
        }

        self.backgrounds = {
            'Black': '\033[40m',
            'Red': '\033[41m',
            'Green': '\033[42m',
            'Yellow': '\033[43m',
            'Blue': '\033[44m',
            'Purple': '\033[45m',
            'Cyan': '\033[46m',
            'White': '\033[47m',
        }

        self.styles = {
            'normal': '\033[0m',
            'bold': '\033[1m',
            'faint': '\033[2m',
            'italic': '\033[3m',
            'underline': '\033[4m',
            'negative': '\033[7m',
            'strikethrough': '\033[9m',
            'end': '\033[0m',
        }

    def __call__(self, text, color: str = 'white', style: str = 'normal') -> str:
        """
        Format the given text with the specified color and style.
        """
        colorCode = self.colors.get(color)
        if colorCode is None:
            raise ValueError(f"Invalid color '{color}'. Available colors are {list(self.colors.keys())}.")

        styleCode = self.styles.get(style)
        if styleCode is None:
            raise ValueError(f"Invalid style '{style}'. Available styles are {list(self.styles.keys())}.")

        return styleCode + colorCode + text + self.styles['end']
