from evalscope.run import run_task
from evalscope.summarizer import Summarizer

def run_eval():
    # 选项 1: python 字典
    # task_cfg = task_cfg_dict

    # 选项 2: yaml 配置文件
    task_cfg = './deploy/evalscope/eval_ablity.yaml'

    # 选项 3: json 配置文件
    # task_cfg = 'eval_openai_api.json'

    run_task(task_cfg=task_cfg)

    print('>> Start to get the report with summarizer ...')
    report_list = Summarizer.get_report_from_cfg(task_cfg)
    print(f'\n>> The report list: {report_list}')

run_eval()