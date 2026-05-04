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

## 4. Baseline com PPO

## 5. Generalização do modelo 5x5 para 10x10

## 6. Curriculum Learning / Transfer Learning

## 7. Ajustes na arquitetura da rede neural

## 8. Possíveis alterações na representação do estado

## 9. Resultados finais

## 10. Análise dos resultados

## 11. Limitações e melhorias futuras

## 12. Conclusão