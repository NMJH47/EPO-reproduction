import os
import socket
from rl_games.common.algo_observer import AlgoObserver

from isaacgymenvs.utils.utils import retry
from isaacgymenvs.utils.reformat import omegaconf_to_dict


class WandbAlgoObserver(AlgoObserver):
    """Need this to propagate the correct experiment name after initialization."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.writer = None

    def before_init(self, base_name, config, experiment_name):
        """
        Must call initialization of Wandb before RL-games summary writer is initialized, otherwise
        sync_tensorboard does not work.
        """

        import wandb

        wandb_unique_id = f"uid_{experiment_name}"
        print(f"Wandb using unique id {wandb_unique_id}")

        cfg = self.cfg

        # this can fail occasionally, so we try a couple more times
        @retry(3, exceptions=(Exception,))
        def init_wandb():
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity or None,
                group=cfg.wandb_group or None,
                tags=cfg.wandb_tags,
                notes=cfg.wandb_notes if hasattr(cfg, 'wandb_notes') else '',
                sync_tensorboard=True,
                id=wandb_unique_id,
                name=experiment_name,
                resume="allow",
            )
       
            wandb.run.log_code(root=cfg.wandb_logcode_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            print('wandb running directory........', wandb.run.dir)

        print('Initializing WandB...')
        init_wandb()

        with open(os.path.join(wandb.run.dir, 'diff.patch'), 'w') as f:
            os.system(f'cd {os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))} && git diff > {f.name}')

        diff_artifact = wandb.Artifact("diff", type="file", description=f"Git diff")
        diff_artifact.add_file(os.path.join(wandb.run.dir, 'diff.patch'))
        wandb.run.log_artifact(diff_artifact)

        if isinstance(self.cfg, dict):
            wandb.config.update(self.cfg, allow_val_change=True)
        else:
            wandb.config.update(omegaconf_to_dict(self.cfg), allow_val_change=True)

    def after_init(self, algo):
        self.writer = algo.writer

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.writer is not None:
            self.writer.flush()

    def finish(self, exit_code=0):
        import wandb

        if self.writer is not None:
            self.writer.flush()
        if wandb.run is not None:
            wandb.finish(exit_code=exit_code)
