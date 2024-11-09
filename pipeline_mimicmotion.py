# Pipeline ClearML

# Import necessary libraries
import os
from clearml import Task, Dataset
from clearml.automation import PipelineController
from dotenv import load_dotenv
import argparse

load_dotenv()

gh_user = os.getenv('GITHUB_USERNAME')
gh_token = os.getenv('GITHUB_TOKEN')


pipe = PipelineController(
     name='pipeline mimicmotion',
     project='MimicMotion Project',
     version='0.0.2',
     add_pipeline_tags=False,
     packages=".MimicMotion/requirements.txt",
)


pipe.set_default_execution_queue("jobs_urgent")


pipe.add_step(
    name='Pipeline mimicmotion',
    base_task_project='MimicMotion',
    base_task_name='Inference v3',
)



pipe.start_locally() # Why this function is active?
