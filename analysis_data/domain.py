"""仅核对输入形状，不生成分析结果。"""
from .io import unique,integer
def summarize(v,root):
    unique(v['dimensions'],'id');integer(v['solver_budget']['max_states'],1)
    if any(not x['states'] for x in v['dimensions']):raise ValueError('边际状态为空')
    return {'dimensions':len(v['dimensions']),'cartesian_states':__import__('math').prod(len(x['states']) for x in v['dimensions']),'thresholds':len(v['thresholds'])}
