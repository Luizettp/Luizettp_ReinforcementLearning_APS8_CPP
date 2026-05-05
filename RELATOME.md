# APS8 — Coverage Path Planning com Reinforcement Learning

## 1. Contextualização

Nesta APS, o objetivo é melhorar o desempenho de um agente de Reinforcement Learning no problema de **Coverage Path Planning (CPP)** em um ambiente `GridWorld`. Diferentemente de um GridWorld tradicional, aqui o agente não deve apenas chegar a um alvo: ele precisa **cobrir todas as células livres do ambiente**, evitando obstáculos, paredes e revisitações desnecessárias.

O principal desafio é que o agente possui **observação parcial**. Ele não recebe o mapa completo do ambiente; toma decisões apenas com base na sua posição, na taxa de cobertura atual e em uma matriz local `3x3` ao seu redor. Assim, a solução precisa melhorar o desempenho sem violar essa restrição.

## 2. Alterações iniciais feitas

Antes de modificar a estratégia de aprendizado, foram feitas correções simples no arquivo `train_grid_world_cpp.py` para tornar os testes mais consistentes. A leitura dos argumentos de terminal foi ajustada para que `train`, `test`, `run` e `curriculum` recebam corretamente os parâmetros necessários, especialmente `max_steps`, evitando erros de execução.

Também foram criadas automaticamente as pastas `data/` e `log/`, corrigido o modo `curriculum` para continuar o treinamento a partir de um modelo pré-treinado sem uma chamada redundante de treino, e alterada a avaliação para usar `deterministic=True` em `test` e `run`. Essas mudanças não alteram o ambiente nem dão acesso ao mapa completo; apenas deixam o pipeline de treino, teste e avaliação mais estável.

## 3. Teste inicial do pipeline

Foi realizado um treino curto apenas para verificar se o pipeline funcionava corretamente:

`python train_grid_world_cpp.py train 5 3 200 10000`

Em seguida, o modelo foi testado em 100 episódios no ambiente `5x5`. O resultado foi:

| Métrica | Valor |
|---|---:|
| Full Coverage Rate | `53.00% (53/100)` |
| Average Coverage | `95.95%` |
| Standard Deviation | `10.06%` |
| Min Coverage | `4.55%` |
| Max Coverage | `100.00%` |
| Average Steps | `161.8` |

Esse resultado ainda não representa a estratégia final, pois o treino teve apenas `10.000` timesteps. Ele serve somente para confirmar que o ambiente, o treinamento, o salvamento do modelo e o teste estão funcionando.

## Tentativa preliminar: enriquecimento da representação do estado

Como a proposta da atividade questionava se faltava algo na representação do estado, foi testada uma modificação simples na observação do agente. A ideia era adicionar informações derivadas da própria trajetória do agente, sem revelar o mapa completo do ambiente.

Foram testadas duas versões: uma com variáveis de memória compacta (`step_ratio`, `revisit_ratio`, `collision_ratio` e `last_action_normalized`) e outra apenas com `step_ratio`. Em ambos os casos, a restrição de observação parcial foi mantida, pois nenhuma informação global sobre o mapa, obstáculos ou células não visitadas foi fornecida ao agente.

Nos testes curtos com `10.000` timesteps, as duas modificações pioraram o desempenho em relação ao estado original. A versão com memória compacta obteve `1/100` episódios com cobertura completa, enquanto a versão apenas com `step_ratio` obteve `0/100`. Por isso, a representação original do ambiente foi mantida para os próximos experimentos, e a estratégia principal passou a focar em arquitetura da rede e curriculum learning.

```markdown
## 4. Estratégia adotada e modificações realizadas

A estratégia inicial foi manter a observação parcial original do ambiente, sem fornecer ao agente o mapa completo. O estado continuou contendo apenas a posição normalizada do agente, a taxa de cobertura e a matriz local `3x3` ao redor dele. Essa decisão foi importante porque a APS exige que o agente tome decisões com base apenas no mapa parcial disponível e em informações coletadas durante a exploração.

A primeira mudança efetiva foi no arquivo `train_grid_world_cpp.py`. O script foi ajustado para receber corretamente os argumentos de `train`, `test`, `run` e `curriculum`, incluindo `max_steps` também nos modos de teste e visualização. Também foram criadas automaticamente as pastas `data/` e `log/`, evitando erros ao salvar modelos e logs. Além disso, o modo `curriculum` foi corrigido para carregar um modelo pré-treinado e continuar o treinamento em outro ambiente sem uma chamada redundante de treino.

Em seguida, foi modificada a arquitetura da rede usada pelo PPO. O modelo continuou usando `MultiInputPolicy`, pois a observação do ambiente é um dicionário com as entradas `"agent"` e `"neighbors"`. Porém, foi adicionada uma arquitetura customizada via `policy_kwargs`, com redes internas maiores para a política (`pi`) e para a função valor (`vf`):

`POLICY_KWARGS = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))`

Com essa alteração, o modelo passou de uma configuração básica:

`PPO("MultiInputPolicy", env, verbose=1, ent_coef=ENTROPY_COEF, device="cpu")`

para uma configuração com maior capacidade de aproximação:

`PPO("MultiInputPolicy", env, verbose=1, ent_coef=ENTROPY_COEF, policy_kwargs=POLICY_KWARGS, device="cpu")`

Essa mudança foi feita porque a proposta da APS apontava que a arquitetura original da rede neural não era adequada. A observação parcial foi preservada; apenas aumentamos a capacidade da rede de aprender a partir dessa observação.

Também foi testado o hiperparâmetro `ENTROPY_COEF`, que controla o incentivo à exploração da política PPO. A configuração inicial usada pelo professor era `0.05`. Após os experimentos, valores menores apresentaram melhor desempenho, especialmente no ambiente `5x5` e na generalização direta para `10x10`.

| Entropy coef | Full Coverage 5x5 | Full Coverage 10x10 direto |
|---:|---:|---:|
| `0.05` | `93/100` | `44/100` |
| `0.02` | `94/100` | `50/100` |
| `0.007` | `95/100` | `54/100` |
| `0.005` | `96/100` | `41/100` |
| `0.00314` | `91/100` | não seguido |

Embora `0.005` tenha sido o melhor no `5x5`, ele piorou a generalização direta para o `10x10`. Por isso, o valor `0.007` foi escolhido como melhor equilíbrio entre bom desempenho no ambiente pequeno e melhor capacidade de transferência para ambientes maiores.

Com `ENTROPY_COEF = 0.007`, arquitetura `[128, 128]` e treinamento de `500.000` timesteps no `5x5`, o agente obteve:

| Métrica | Resultado |
|---|---:|
| Full Coverage Rate | `95.00% (95/100)` |
| Average Coverage | `99.77%` |
| Min Coverage | `95.45%` |
| Max Coverage | `100.00%` |
| Average Steps | `42.4` |

Depois, esse modelo foi testado diretamente no ambiente `10x10`, com `12` obstáculos e `400` passos máximos, obtendo:

| Métrica | Resultado |
|---|---:|
| Full Coverage Rate | `54.00% (54/100)` |
| Average Coverage | `97.70%` |
| Min Coverage | `53.41%` |
| Max Coverage | `100.00%` |
| Average Steps | `287.3` |

Esse resultado mostrou que a arquitetura melhorada e o ajuste de entropia ajudam, mas ainda não são suficientes para generalizar totalmente do `5x5` para o `10x10`.

Por isso, foi aplicado `curriculum learning`: o modelo treinado no `5x5` foi carregado e continuou treinando no ambiente `10x10`. Com `ENTROPY_COEF = 0.007`, o resultado no `10x10` após curriculum foi:

| Métrica | Resultado |
|---|---:|
| Full Coverage Rate | `76.00% (76/100)` |
| Average Coverage | `98.91%` |
| Min Coverage | `73.86%` |
| Max Coverage | `100.00%` |
| Average Steps | `230.3` |

Esse resultado confirmou que o curriculum learning melhora a adaptação ao ambiente maior, mas ainda não alcança cobertura próxima de `100%`.

## 5. Próxima etapa: curriculum intermediário com ambiente 8x8

Como o salto direto de `5x5` para `10x10` ainda se mostrou difícil, a próxima estratégia será usar um ambiente intermediário `8x8`. Não é necessário criar um novo arquivo de ambiente para isso, pois o `GridWorldCPPEnv` já recebe o tamanho do grid como parâmetro:

`def __init__(self, render_mode=None, size: int = 5, obs_quantity: int = 3, max_steps: int = 200):`

Assim, o ambiente `8x8` é criado dinamicamente pelo próprio script de treino, usando os argumentos enviados pelo terminal.

A proporção de obstáculos segue a lógica da configuração do professor:

| Ambiente | Células | Obstáculos | Proporção aproximada |
|---:|---:|---:|---:|
| `5x5` | `25` | `3` | `12%` |
| `10x10` | `100` | `12` | `12%` |
| `20x20` | `400` | `48` | `12%` |

Para o `8x8`, usamos então:

`8x8 → 64 células → 8 obstáculos`

O limite de passos também segue a proporção usada nos ambientes principais:

`max_steps = 40 × dimensão`

Portanto:

`8x8 → 320 passos máximos`

A próxima sequência de treino será:

`python train_grid_world_cpp.py curriculum 8 8 320 500000`

usando como ponto de partida o modelo `5x5` treinado com `ENTROPY_COEF = 0.007`.

Depois, o modelo treinado no `8x8` será testado no próprio `8x8`:

`python train_grid_world_cpp.py test 8 8 320`

e também no `10x10`:

`python train_grid_world_cpp.py test 10 12 400`

A hipótese é que o curriculum intermediário `5x5 → 8x8 → 10x10` seja mais eficiente do que o salto direto `5x5 → 10x10`, pois o agente passa por uma dificuldade intermediária antes de ser treinado no ambiente final.
```




## X. Baseline com PPO

## Y. Generalização do modelo 5x5 para 10x10

## Z. Curriculum Learning / Transfer Learning

## X. Ajustes na arquitetura da rede neural

## Y. Possíveis alterações na representação do estado

## Z. Resultados finais

## X. Análise dos resultados

## Y. Limitações e melhorias futuras

## Z. Conclusão