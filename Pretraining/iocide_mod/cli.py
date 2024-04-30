"""
Command-line interface
"""
import argparse
import io
import logging
from pathlib import Path
from pkg_resources import get_distribution
import sys
import re
from tqdm import tqdm
from multiprocessing import Process, cpu_count, Manager
from collections import defaultdict
from . import blobs, deobfuscate, email, hashes, hostname, ip, url


logger = logging.getLogger(name=__name__)


REFANGING_MODULES = [url, email, ip]

RE_MD5 = re.compile(r'(?i)(?<![a-z0-9])[a-f0-9]{32}(?![a-z0-9])')
RE_SHA = re.compile(r'(?i)(?<![a-z0-9])[a-f0-9]{64}(?![a-z0-9])')
RE_BTC_ADDR = re.compile(r'(?<![a-zA-Z0-9])([13]|bc1)[A-HJ-NP-Za-km-z1-9]{27,34}(?![a-zA-Z0-9])')
RE_CVE = re.compile(r'(?i)(CVE-(1999|2\d{3})-(0\d{2}[0-9]|[1-9]\d{3,}))')

def replace_certain_iocs(text, replace_these):
	# Since urls are a superset, they must be last
	do_urls = 'url' in replace_these
	
	tq = tqdm(replace_these, desc="Processing type ")
	for type in tq:
		tq.set_description(f"Processing type {type}")
		if type == 'url':
			continue
		if type == 'ip':
			text = ip.replace(text, "<IP>")
		if type == 'email':
			text = email.replace(text, "<EMAIL>")
		if type == 'MD5':
			text = RE_MD5.sub("<MD5>", text)
		if type == 'SHA':
			text = RE_MD5.sub("<SHA>", text)
		if type == 'BTC':
			text = RE_BTC_ADDR.sub("<BTC>", text)
		if type == 'CVE':
			text = RE_CVE.sub("<CVE>", text)
	if do_urls:
		text = url.replace(text, "<URL>")		
	return text

def extract_url_md5(result, lines, refang=False, skip_these=[]):
	for text, span in lines:
		res = {}
      
		mname = url.__name__.split(".")[-1]
		if mname in skip_these:
			continue
		# logger.debug("Extracting ", mname)
		res[mname] = [a for a in url.extract(text=text, refang=refang)]
		if 'MD5' not in skip_these:
			# logger.debug("Extracting ", 'MD5')
			res['MD5'] = [(str(a), a.span()) for a in RE_MD5.finditer(text)]
		result.append((res, span))
	print("finished f1")
  
def extract_email_sha(result, lines, refang=False, skip_these=[]):
	for text, span in lines:
		res = {}
  
		mname = email.__name__.split(".")[-1]
		if mname in skip_these:
			continue
		# logger.debug("Extracting ", mname)
		res[mname] = [a for a in email.extract(text=text, refang=refang)]  
		if 'SHA' not in skip_these:
			# logger.debug("Extracting ", 'SHA')
			res['SHA'] = [(str(a), a.span()) for a in RE_SHA.finditer(text)]
		result.append((res, span))
	print("finished f2")
	
 
def extract_ip_btc_cve(result, lines, refang=False, skip_these=[]):
	for text, span in lines:
		res = {}
  
		mname = ip.__name__.split(".")[-1]
		if mname in skip_these:
			continue
		# logger.debug("Extracting ", mname)
		res[mname] = [a for a in ip.extract(text=text, refang=refang)]  			
		if 'BTC' not in skip_these:
			# logger.debug("Extracting ", 'BTC')
			res['BTC'] = [(str(a), a.span()) for a in RE_BTC_ADDR.finditer(text)]
		if 'CVE' not in skip_these:
			# logger.debug("Extracting ", 'CVE')
			res['CVE'] = [(str(a), a.span()) for a in RE_CVE.finditer(text)]
		result.append((res, span))
	print("finished f3")


def extract_all_multi(text, refang=False, skip_these=['']):
	"""
 	extract_all_modified with multiprocessing
	"""
	manager = Manager()
	result = manager.list()
    
	lines = text.split('\n')
	chunk_size = len(lines) // cpu_count()
    
	chunks = []
	initial_idx = 0
	for i in range(0, len(lines), chunk_size):
		_text = ' '.join(lines[i:i+chunk_size])
		start_idx = initial_idx
		end_idx = start_idx + len(_text)
		chunks.append((_text, (start_idx, end_idx)))
		initial_idx = end_idx + 1
        
	p1 = Process(target=extract_url_md5, args=(result, chunks, refang, skip_these))
	p2 = Process(target=extract_email_sha, args=(result, chunks, refang, skip_these))
	p3 = Process(target=extract_ip_btc_cve, args=(result, chunks, refang, skip_these))
    
	processes = [p1, p2, p3]
	for process in processes:
		process.start()
	for process in processes:
		process.join()

	print("All processes completed")
    
	res_dict = defaultdict(list)
	for res, span in result:
		for key, val in res.items():
			temp = [(v[0], (span[0] + v[1][0], span[0] + v[1][1])) for v in val] if val else val
			if temp:
				res_dict[key].extend(temp)
	return res_dict


def extract_all_modified(text, refang=False, skip_these=['']):
	"""
	Extract all known IOC types, binary blobs, and binary-embedded text
	"""
	res = {}
	for module in REFANGING_MODULES:
		# mname = module.__name__.split(".")[1]
		mname = module.__name__.split(".")[-1]
		if mname in skip_these:
			continue
		# logger.debug("Extracting ", mname)
		res[mname] = [a for a in tqdm(module.extract(text=text, refang=refang))]
	if 'MD5' not in skip_these:
		res['MD5'] = [(str(a), a.span()) for a in tqdm(RE_MD5.finditer(text))]
	if 'SHA' not in skip_these:
		res['SHA'] = [(str(a), a.span()) for a in tqdm(RE_SHA.finditer(text))]
	if 'BTC' not in skip_these:
		res['BTC'] = [(str(a), a.span()) for a in tqdm(RE_BTC_ADDR.finditer(text))]
	if 'CVE' not in skip_these:
		res['CVE'] = [(str(a), a.span()) for a in tqdm(RE_CVE.finditer(text))]
	return res


def main():
	package_name, *_ = __name__.split('.', 1)
	package_version = get_distribution(package_name).version
	root_parser = argparse.ArgumentParser(
		description='Indicator of Compromise (IOC) Detection')

	root_parser.add_argument(
		'-V', '--version',
		action='version',
		version=f'%(prog)s {package_version}',
	)
	root_parser.add_argument(
		'-l', '--log-level', default='WARNING', help='Set the log level')
	root_parser.add_argument(
		'-r', '--refang', action='store_true', help='Refang detected IOCs')
	root_parser.add_argument(
		'--raw',
		action='store_true',
		help="Don't normalise input text before scanning for IOCs",
	)
	root_parser.add_argument('-i', '--input', type=Path, help='Input file')
	root_parser.add_argument(
		'--limit', type=int, help='Embedded binary text search recursion limit')

	root_parser.set_defaults(function=extract_all)

	subparsers = root_parser.add_subparsers()
	
	all_parser = subparsers.add_parser(
		'all', description='Extract all IOC types')
	all_parser.set_defaults(function=extract_all)

	blobs_parser = subparsers.add_parser(
		'blobs', description='Extract embedded binary blobs')
	blobs_parser.set_defaults(function=blobs.extract)

	email_parser = subparsers.add_parser(
		'email', description='Extract email addresses')
	email_parser.set_defaults(function=email.extract)

	hashes_parser = subparsers.add_parser(
		'hashes', description='Extract hash values')
	hashes_parser.set_defaults(function=hashes.extract)

	hostname_parser = subparsers.add_parser(
		'hostname', description='Extract hostnames')
	hostname_parser.set_defaults(function=hostname.extract)

	ip_parser = subparsers.add_parser('ip', description='Extract IP addresses')
	ip_parser.set_defaults(function=ip.extract)

	url_parser = subparsers.add_parser('url', description='Extract URLs')
	url_parser.set_defaults(function=url.extract)

	secrets_parser = subparsers.add_parser(
		'secrets', description='Extract text from embedded binary blobs')
	secrets_parser.set_defaults(function=blobs.extract_text)
	secrets_parser.add_argument(
		'--raw-secrets',
		action='store_true',
		help="Don't normalise embedded binary text before recursive search",
	)

	normalise_parser = subparsers.add_parser(
		'normalise', description='Output the normalised input text')
	normalise_parser.set_defaults(function=deobfuscate.normalise)

	namespace = root_parser.parse_args(sys.argv[1:])
	arguments = vars(namespace)

	log_level = arguments.pop('log_level')
	logging.basicConfig(level=log_level.upper())
	logger.debug('parsed args: %r', namespace)

	if namespace.input is None:
		in_file = io.BytesIO(sys.stdin.buffer.read())
	else:
		in_file = namespace.input.open('rb')

	out_file = sys.stdout
	command_function = namespace.function
	normalise = not namespace.raw

	if command_function is blobs.extract_text:
		out_file.writelines(
			command_function(data=in_file, embedded_only=True, depth=None))
		return

	for text in blobs.extract_text(
			data=in_file, depth=namespace.limit, normalise=normalise):
		logger.debug(text)
		if command_function is deobfuscate.normalise:
			out_file.write(command_function(text=text))
			return

		optionals = {'refang': namespace.refang}
		if command_function is hashes.extract:
			optionals.pop('refang')

		out_values = command_function(text=text, **optionals)
		out_file.writelines(f'{v}\n' for v in out_values)