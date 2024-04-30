"""
Defanged/Obfuscated Indicator of Compromise (IOC) Detection

Use `normalise` and `extract_all` to emulate cli invocation without arguments.

Extraction of each specific IOC type is implemented in the corresponding
module.
"""
from . import blobs, email, hashes, hostname, ip, url
from .cli import extract_all_modified, replace_certain_iocs, extract_all_multi
from .deobfuscate import normalise
