# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""
from pathlib import Path

from kedro.pipeline import Pipeline, node
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
from kedro.runner import run_node
from kedro_dask_example.io import maybe_worker_client
from dask.distributed import Client, LocalCluster, wait

from .nodes import predict, report_accuracy, train_model, split_data


def mini_pipeline(name, parameters):
    return [
        node(
            train_model,
            ["example_train_x", "example_train_y", parameters],
            f"example_model_{name}",
            name=f"train_{name}"
        ),
        node(
            predict,
            dict(model=f"example_model_{name}", test_x="example_test_x"),
            f"example_predictions_{name}",
            name=f"predict_{name}"
        ),
        node(
            report_accuracy,
            [f"example_predictions_{name}", "example_test_y"],
            None,
            name=f"report_{name}"),
    ]


def create_pipeline(**kwargs):
    nodes = [
        node(
            split_data,
            ["example_iris_data", "params:example_test_data_ratio"],
            dict(
                train_x="example_train_x",
                train_y="example_train_y",
                test_x="example_test_x",
                test_y="example_test_y",
            ),
            name="split"
        ),
    ]
    nodes += mini_pipeline("a", "params:a")
    nodes += mini_pipeline("b", "params:b")
    nodes += mini_pipeline("c", "params:c")
    return Pipeline(nodes)


def node_wrapper(node, catalog, *args, **kwargs):
    with maybe_worker_client() as c:
        run_node(node, catalog)
    return None


def get_node_future(client, futures, node, catalog, dependencies):
    if node._unique_key in futures:
        return futures[node._unique_key]
    deps = dependencies[node]
    dep_futures = [get_node_future(client, futures, x,
                                   catalog, dependencies) for x in deps]
    fut = client.submit(node_wrapper, node, catalog, *dep_futures)
    futures[node._unique_key] = fut
    return fut


def run():
    project_path = Path.cwd()
    metadata = bootstrap_project(project_path)
    env = None
    session = KedroSession.create(project_path=project_path, env=env)
    context = session.load_context()
    catalog = context.catalog
    pipeline = create_pipeline()
    # cluster = LocalCluster(processes=False, threads_per_worker=1)
    client = Client("tcp://127.0.0.1:8786")
    futures = {}
    for node, parent_nodes in pipeline.node_dependencies.items():
        get_node_future(client, futures, node, catalog, pipeline.node_dependencies)
    wait(futures)
