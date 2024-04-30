# Text types to be replaced or classified
SEMILINGUISTIC_TEXT = ['url', 'email']
NONLINGUISTIC_TEXT = ['ip', 'MD5', 'SHA', 'BTC', 'CVE']



# Experiment settings
"""
REPLACE_ALL = {
    'replace': SEMILINGUISTIC_TEXT + NONLINGUISTIC_TEXT,
    'dont_mask': [] 
    } No classification is done with this setting, so usual MLM model should be used
"""
MLM_ON_ALL = {
    'replace': [],
    'dont_mask': []
    }
ONLY_LING_MLM_W_REPLACE = {
    'replace': NONLINGUISTIC_TEXT,
    'dont_mask': SEMILINGUISTIC_TEXT
    }
SEMI_LING_MLM_W_REPLACE = {
    'replace': NONLINGUISTIC_TEXT,
    'dont_mask': []
    }
SEMI_LING_MLM = {
    'replace': [],
    'dont_mask': NONLINGUISTIC_TEXT
    }
ONLY_LING_MLM = {
    'replace': [],
    'dont_mask': NONLINGUISTIC_TEXT+SEMILINGUISTIC_TEXT
    }

Experiment_modes = {
    'mlm_on_all': MLM_ON_ALL,
    'only_ling_mlm_w_replace': ONLY_LING_MLM_W_REPLACE,
    'semi_ling_mlm_w_replace': SEMI_LING_MLM_W_REPLACE,
    'semi_ling_mlm': SEMI_LING_MLM,
    'semi_ling_no_tok': SEMI_LING_MLM,
    'only_ling_mlm': ONLY_LING_MLM,
    'replace_all': 'replace_all',
    'og_run': 'og_run'
}
