from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'admin',
    'start_date': days_ago(1),
    'retries': 5,
    #"retry_delay": datetime.timedelta(minutes=5),
    'task_concurrency': 1
}

pipelines = {'train': {'schedule': '30 * * * *'},
             'predict': {'schedule': '40 * * * *'}}


def init_dag(dag, task_id):
    with dag:
        t1 = BashOperator(
            task_id=f'{task_id}',
            bash_command=f'python /home/primera7790/ml_projects/linux_or_windows/{task_id}.py')
    return dag


for task_id, params in pipelines.items():
    dag = DAG(task_id,
              schedule_interval=params['schedule'],
              max_active_runs=1,
              default_args=default_args
              )
    init_dag(dag, task_id)
    globals()[task_id] = dag