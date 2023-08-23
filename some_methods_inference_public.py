import click
from tqdm import tqdm
import gc
from road_anomaly_benchmark.evaluation import Evaluation
from methods import baselines as baselines_module

METHOD_KEYS = {
    "Mahalanobis": "O5AMylqoSaq5Wjc",
    "Max_softmax": "WVFTc4ka37xASZV",
    "ODIN": "WVFTc4ka37xASZV",
    "Entropy_max": "kCgnr0LQuTbrArA",
    "SynBoost": "0",
}

def name_list(name_list):
    return [name for name in name_list.split(',') if name]

@click.command()
@click.option('--methods', default='ALL', help=f'Which methods to run, ALL or some of {list(METHOD_KEYS.keys())}')
@click.option('--dsets', default='ObstacleTrack-validation', help='List of datasets to evaluate on, ex: ObstacleTrack-validation,LostAndFound-testNoKnown')

def main(methods, dsets):
    if methods == 'ALL':
        methods = list(METHOD_KEYS.keys())
    else:
        methods = name_list(methods)
    dataset_names = name_list(dsets)

    for methodname in methods:
        modelid = METHOD_KEYS[methodname]
        method_object = getattr(baselines_module, methodname)
        method = method_object(modelid)

        for dset in dataset_names:
            print(f"-- Inference: Method {methodname} on Dataset {dset} --")

            ev = Evaluation(
                method_name = methodname,
                dataset_name = dset,
            )

            for frame in tqdm(ev.get_frames(), total=ev.__len__()):
                result = method.anomaly_score(frame.image)
                ev.save_output(frame, result)

            ev.wait_to_finish_saving()

        del method
        gc.collect()


if __name__ == '__main__':
    main()
