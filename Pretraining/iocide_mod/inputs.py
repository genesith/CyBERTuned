"""
File contents extraction
"""
import io
from logging import getLogger
from zipfile import BadZipFile, ZipFile

import chardet
import pdfminer.high_level
from pdfminer.pdfparser import PDFSyntaxError


logger = getLogger(name=__name__)


class EncodingDetectionFailure(Exception):
	"""
	Automatic detection of text encoding in binary data has failed
	"""
	def __init__(self, *args, encoding=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.encoding = encoding


def unpack_files(file):
	"""
	Iterate through files packed into the specified file
	"""
	try:
		zip = ZipFile(file)
	except BadZipFile:
		file.seek(0)
		yield file
		return

	try:
		for info in zip.infolist():
			with zip.open(info) as member_file:
				yield member_file
	finally:
		zip.close()


def decode_file(file, required_confidence=0.5, fallback='utf_8'):
	"""
	Get the text content of a file, detecting the filetype and text encoding
	where necessary
	"""
	detector = chardet.UniversalDetector()
	position = file.tell()
	try:
		return pdfminer.high_level.extract_text(pdf_file=file)
	except PDFSyntaxError:
		file.seek(position)

	for line in file:
		detector.feed(line)
		if detector.done:
			break
	
	detector.close()
	file.seek(position)
	encoding = detector.result['encoding']
	try:
		if encoding is None:
			raise EncodingDetectionFailure('Failed to detect encoding')

		confidence = detector.result['confidence']
		if confidence < required_confidence:
			raise EncodingDetectionFailure(
				f'Insufficient confidence in detected encoding {encoding}:'
				f' {confidence} < {required_confidence}',
				encoding=encoding,
			)

		wrapper = io.TextIOWrapper(buffer=file, encoding=encoding)
		text = wrapper.read()
		return text
	except (EncodingDetectionFailure, UnicodeDecodeError) as error:
		if fallback is None:
			raise

		logger.debug(error, exc_info=error)

	file.seek(position)
	logger.debug('Falling back to %r encoding', fallback)
	return io.TextIOWrapper(buffer=file, encoding=fallback).read()
