# Modelos 20x20 preservados

Todos usam observação parcial local e o ambiente `20x20_grid_world_cpp.py`.
O script `20x20_train_grid_world_cpp.py` aceita `CPP_PROFILE=gladiador`, `CPP_PROFILE=merlin` ou `CPP_PROFILE=imperador`. Sem `CPP_PROFILE`, o padrão experimental é `merlin`.

| Modelo | Arquivo | Estado | Reward | Observação | Uso |
|---|---|---|---|---|---|
| `Gladiador_2020` | `data/20x20_experiments/Gladiador_2020.zip` | `local_history_v2` | `original` | `3x3` | Campeão anterior, linha mais conservadora. |
| `MERLIN_2020` | `data/20x20_experiments/MERLIN_2020.zip` | `local_memory_decay_v2` | `late_finish_v1` | `3x3` | Rastro direcional curto, sem mapa e sem lista de coordenadas. |
| `Imperador_20X20` | `data/20x20_experiments/Imperador_20X20.zip` | `local_memory_v1` | `original` | `3x3` | Memória de células livres já observadas e ainda não visitadas. |

## Reproduzir testes

Gladiador:

```powershell
$env:CPP_PROFILE='gladiador'
$env:CPP_STATE_VARIANT='local_history_v2'
$env:CPP_REWARD_VARIANT='original'
$env:CPP_LOCAL_VIEW_SIZE='3'
.\venv\Scripts\python.exe 20x20_train_grid_world_cpp.py test 20 48 2400 1000 data\20x20_experiments\Gladiador_2020.zip
```

MERLIN:

```powershell
$env:CPP_PROFILE='merlin'
$env:CPP_STATE_VARIANT='local_memory_decay_v2'
$env:CPP_REWARD_VARIANT='late_finish_v1'
$env:CPP_LOCAL_VIEW_SIZE='3'
$env:CPP_MEMORY_DECAY='0.92'
$env:CPP_MEMORY_TRACE_GAIN='0.40'
.\venv\Scripts\python.exe 20x20_train_grid_world_cpp.py test 20 48 1500 1000 data\20x20_experiments\MERLIN_2020.zip
```

Imperador:

```powershell
$env:CPP_PROFILE='imperador'
$env:CPP_STATE_VARIANT='local_memory_v1'
$env:CPP_REWARD_VARIANT='original'
$env:CPP_LOCAL_VIEW_SIZE='3'
.\venv\Scripts\python.exe 20x20_train_grid_world_cpp.py test 20 48 1000 1000 data\20x20_experiments\Imperador_20X20.zip
```
