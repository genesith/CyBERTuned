CYNER_SETTINGS = {
    'learning_rate':5e-5,
    'total_batch_size':32,
}
CYSECED_SETTINGS = {
    'learning_rate':5e-5,
    'total_batch_size':1,
}
MTDB_N_SETTINGS = {
    'learning_rate':5e-5,
    'total_batch_size':8,
    'accumulate': 2
}
MTDB_SETTINGS = CYNER_SETTINGS
CYDEC_SETTINGS = CYNER_SETTINGS
TT_SETTINGS = CYNER_SETTINGS

CASIE_SETTINGS = CYSECED_SETTINGS

ALL_SETTINGS = {
    'CyNER': CYNER_SETTINGS,
    'CySecED': CYSECED_SETTINGS,
    'MalwareTextDB_T1': MTDB_SETTINGS,
    'MalwareTextDB_New': MTDB_N_SETTINGS,
    'CASIE_T1': CASIE_SETTINGS,
    'CYDEC': CYDEC_SETTINGS,
    'TwitterThreats_T1': TT_SETTINGS
}