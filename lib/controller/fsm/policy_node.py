import threading
import typing as tp
import torch
from lib.deploy.fsm.policy_node import PolicyNode as BasePolicyNode

class PolicyNode(BasePolicyNode):
    @torch.no_grad()
    @torch.inference_mode()
    def _warmup(self):
        carries = self.processor.init_carries()
        obs, _, _ = self._get_observation(carries, 0, dry_run=True)
        for _ in range(4):
            self.policy(obs)
        self._reset()

    def _get_observation(self, carries, timestep: int, dry_run: bool = False) -> torch.Tensor:
        obs = self.env.get_observation()
        action_carry, observation_carry = carries
        proc_obs, observation_carry = self.processor.process_obs(obs, timestep, action_carry, observation_carry, dry_run=dry_run)
        return proc_obs, obs, (action_carry, observation_carry)

    def deactivate(self):
        if self.processor.is_debug or self.processor.record_vr:
            t = threading.Thread(
                target=self.processor.save_vr_log,
                args=(self.name,),
                daemon=True
            )
            t.start()

        return super().deactivate()

    def get_info(self) -> tp.Dict[str, tp.Any]:
        return {'processor_info': self.processor.get_info()}
