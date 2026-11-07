from types import SimpleNamespace

import pytest
import torch

from distil_trainer import MemoryEfficientSyncRefModelCallback
from ssd.experiment_timing import verify_teacher_ema_updates


def test_ema_callback_updates_teacher_and_records_optimizer_boundaries(tmp_path):
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=True)
    student = torch.nn.Linear(2, 1, bias=False)
    teacher = torch.nn.Linear(2, 1, bias=False).requires_grad_(False)
    with torch.no_grad():
        student.weight.fill_(3)
        teacher.weight.fill_(1)
    callback = MemoryEfficientSyncRefModelCallback(teacher, accelerator)
    args = SimpleNamespace(ref_model_sync_steps=2, ref_model_mixup_alpha=0.01,
                           sync_ref_model=True, output_dir=str(tmp_path))
    for step in (1, 2, 3, 4):
        callback.on_step_end(args, SimpleNamespace(global_step=step), None, model=student)
    torch.testing.assert_close(teacher.weight, torch.full_like(teacher.weight, 1.0398))
    trainer = SimpleNamespace(args=args, state=SimpleNamespace(global_step=4),
                              ref_model=teacher,
                              optimizer=SimpleNamespace(param_groups=[{"params": list(student.parameters())}]),
                              callback_handler=SimpleNamespace(callbacks=[callback]))
    assert verify_teacher_ema_updates(trainer)['updates'] == 2
    trainer.optimizer.param_groups[0]['params'].append(teacher.weight)
    with pytest.raises(ValueError, match='outside the optimizer'):
        verify_teacher_ema_updates(trainer)
    trainer.optimizer.param_groups[0]['params'].pop()
    callback.ema_update_steps.pop()
    with pytest.raises(ValueError, match='optimizer-step boundaries'):
        verify_teacher_ema_updates(trainer)
