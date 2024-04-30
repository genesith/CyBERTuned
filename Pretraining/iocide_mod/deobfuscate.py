"""
Deobfuscation of input text
"""
from unidecode import unidecode


def normalise(text: str):
	"""
	Strip zero-width spaces and convert non-ASCII characters to ASCII where
	possible
	"""
	visibile_text = text.replace('\u200b', '')
	return unidecode(visibile_text, errors='preserve')
