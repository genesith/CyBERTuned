"""
Embedded binary detection
"""
import io
from logging import getLogger


from . import deobfuscate, rfc_3548
from .inputs import decode_file, unpack_files


logger = getLogger(name=__name__)


def extract(text, refang=False):
	"""
	Generate detected binary blobs
	"""
	yield from rfc_3548.extract(text, refang=refang)


def extract_text(
		data,
		skip_failures=True,
		depth=1,
		normalise=True,
		embedded_only=False,
		**decode_kwargs,
):
	"""
	Recursively generate decoded text strings from binary data
	"""
	if depth is not None and depth <= 0:
		return

	if isinstance(data, bytes):
		data = io.BytesIO(data)
	
	for packed_file in unpack_files(data):
		try:
			text = decode_file(file=packed_file, **decode_kwargs)
		except UnicodeDecodeError as error:
			if not skip_failures:
				raise

			logger.debug(error, exc_info=error)
			logger.debug('Skipping failure')
			continue

		if normalise:
			searchable_text = deobfuscate.normalise(text)
		else:
			searchable_text = text
		
		if not embedded_only:
			yield searchable_text

		next_depth = None if depth is None else depth - 1
		if next_depth == 0:
			continue

		for blob in extract(text=searchable_text, refang=True):
			yield from extract_text(
				data=blob,
				skip_failures=True,
				depth=next_depth,
				normalise=normalise,
				embedded_only=False,
				**decode_kwargs,
			)
