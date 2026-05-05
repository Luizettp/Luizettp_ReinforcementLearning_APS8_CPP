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

## 4. Estratégia inicial: ajuste da arquitetura da `MultiInputPolicy`

A primeira estratégia efetivamente adotada foi manter a representação original do estado e melhorar a arquitetura da rede neural usada pelo PPO. A observação do ambiente foi preservada exatamente dentro da lógica de observação parcial: o agente continua recebendo apenas sua posição normalizada, a taxa de cobertura atual e a matriz local `3x3` ao seu redor. Assim, não foi fornecido ao agente o mapa completo, a posição global dos obstáculos ou a lista de células ainda não visitadas.

A mudança principal foi feita no arquivo `train_grid_world_cpp.py`. O modelo continuou usando `MultiInputPolicy`, pois a observação do ambiente é um dicionário com as entradas `"agent"` e `"neighbors"`. Porém, em vez de usar a arquitetura padrão da Stable-Baselines3, foi definida uma arquitetura maior para as redes internas de política (`pi`) e função valor (`vf`):

`pi = [128, 128]`

`vf = [128, 128]`

Com isso, a criação do modelo passou a incluir `policy_kwargs`, aumentando a capacidade da rede de aprender padrões de cobertura, evitar revisitações excessivas e interpretar melhor a matriz local de vizinhança, sem alterar a informação disponível ao agente.

Essa alteração responde diretamente ao ponto levantado na proposta da APS: a arquitetura padrão usada no treinamento inicial não parecia adequada para obter cobertura próxima de `100%`. A hipótese testada foi que o problema não estava necessariamente em dar mais informação global ao agente, mas em dar maior capacidade à rede para aprender uma política melhor a partir da observação parcial já existente.

Com essa modificação, foi realizado um treinamento no ambiente `5x5`, com `3` obstáculos, `200` passos máximos por episódio e `500.000` timesteps. O resultado obtido no teste com `100` episódios foi:

| Métrica | Resultado |
|---|---:|
| Full Coverage Rate | `93.00% (93/100)` |
| Average Coverage | `99.64%` |
| Standard Deviation | `1.39%` |
| Min Coverage | `90.91%` |
| Max Coverage | `100.00%` |
| Average Steps | `55.1` |

Esse resultado já representa uma melhora expressiva em relação aos valores de referência informados pelo professor para o ambiente `5x5`, que variavam entre `69/100` e `81/100` episódios com cobertura completa. A nova arquitetura atingiu `93/100`, além de uma cobertura média de `99.64%`, indicando que mesmo os episódios sem cobertura completa ficaram muito próximos de cobrir todo o ambiente.

Portanto, a primeira conclusão experimental é que a melhoria da arquitetura da `MultiInputPolicy` teve impacto positivo claro no desempenho do agente, sem violar a restrição de observação parcial do ambiente.

## X. Baseline com PPO

## Y. Generalização do modelo 5x5 para 10x10

## Z. Curriculum Learning / Transfer Learning

## X. Ajustes na arquitetura da rede neural

## Y. Possíveis alterações na representação do estado

## Z. Resultados finais

## X. Análise dos resultados

## Y. Limitações e melhorias futuras

## Z. Conclusão