# APS5 — Deep Reinforcement Learning no MountainCar-v0 -- Luizettp

## 1. Contextualização

Nesta atividade, comparamos duas abordagens baseadas em valor no ambiente **`MountainCar-v0`**:

- **Q-Learning tabular com discretização do espaço de estados**;
- **Deep Q-Learning (DQN)** com aproximação da função `Q(s,a)` por meio de uma rede neural.

O `MountainCar` é um ambiente em que um carro deve sair de um vale e atingir o topo da montanha. O estado é contínuo e possui duas variáveis:

- **posição**;
- **velocidade**.

As ações continuam discretas:

- `0`: acelera para a esquerda  
- `1`: não acelera  
- `2`: acelera para a direita  

No caso do **Q-Learning tabular**, como a tabela `Q(s,a)` exige estados discretos, foi necessário discretizar posição e velocidade exatamente como sugerido no handout. Já no **Deep Q-Learning**, a Q-table foi substituída por uma rede neural, cuja entrada é o estado contínuo e cuja saída é um vetor com o valor estimado de cada ação.

Nesta APS, o **Q-Learning tabular** foi mantido como baseline comparativo. Isso foi importante porque permitiu observar, no mesmo ambiente, a diferença entre uma solução clássica baseada em tabela e uma solução de **Deep Reinforcement Learning**. Como o `MountainCar-v0` possui estado contínuo, essa comparação faz bastante sentido: no método tabular, foi necessário adaptar o problema por discretização; no método profundo, a aproximação da função valor ficou a cargo da rede neural.

Na implementação final do DQN, foi utilizado **PyTorch**, o que tornou o treinamento mais viável na prática e permitiu uma implementação mais estável para o ambiente.

### Hiperparâmetros principais

#### Q-Learning tabular

Os hiperparâmetros usados no baseline tabular foram os mesmos adotados como referência da APS 4:

```python
alpha = 0.1
gamma = 0.99

epsilon_init_tabular = 0.7
epsilon_min_tabular = 0.01
epsilon_dec_tabular = 0.999

episodes_tabular = 15000
n_runs_tabular = 10
max_steps_tabular = 200
```

#### Deep Q-Learning

A configuração final escolhida para o Deep Q-Learning foi:

```python
gamma = 0.99
epsilon_init = 1.0
epsilon_min = 0.01
epsilon_dec = 0.995

episodes_dqn = 500
n_runs_dqn = 5
max_steps_dqn = 200

batch_size = 64
memory_size = 10000
learning_rate = 0.001

hidden_1 = 128
hidden_2 = 128
target_update_freq = 5
```

---

## Atividade proposta

### 1. Treine o agente usando Q-Learning para o ambiente MountainCar usando os melhores hiperparâmetros encontrados na atividade anterior.

Nesta etapa, foi reutilizada a implementação de **Q-Learning tabular** da APS 4. Como o estado do `MountainCar-v0` é contínuo, posição e velocidade foram discretizadas usando os mesmos fatores indicados no handout, construindo uma **Q-table 3D** com a estrutura:

- posição discretizada;
- velocidade discretizada;
- ação.

A escolha por usar o Q-Learning tabular como ponto de partida foi importante porque ele fornece uma referência direta clássica a ser comparada com o Deep Q-Learning.

Os resultados iniciais observados foram:

- **Melhor run escolhida:** `1`
- **Reward médio nos 100 primeiros episódios (média entre runs):** `-200.0`
- **Reward médio nos 100 últimos episódios (média entre runs):** `-171.865`

O agente tabular partiu do pior caso possível, com episódios terminando sistematicamente em `-200`, e conseguiu evoluir para uma média final bem melhor. Isso mostra que, mesmo com a limitação da discretização, o método foi capaz de aprender a lógica básica do ambiente, especialmente a necessidade de ganhar embalo para alcançar o topo.

### 2. Execute o treinamento n vezes onde este n precisa ser maior ou igual a 5.

Para reduzir a influência de uma única execução e tornar a análise mais robusta, o treinamento do **Q-Learning tabular** foi repetido **10 vezes**.

Essa repetição foi importante porque o processo de aprendizado em RL depende de fatores aleatórios, como:

- escolha inicial de ações;
- exploração ao longo do tempo;
- trajetórias específicas observadas pelo agente.

Ao rodar várias execuções independentes, foi possível chegarmos, após MUITO tempo esperando, em:

- **Número de runs:** `10`
- **Melhor run escolhida:** `1`
- **Reward médio final entre runs:** `-171.865`

### 3. Colete todos os dados para criar a curva de aprendizado.

Os rewards acumulados por episódio foram armazenados em todas as execuções do **Q-Learning tabular**. A partir disso, foram calculados:

- reward por episódio de cada run;
- média por episódio entre runs;
- desvio-padrão entre runs;
- curva suavizada por *rolling average*.

Esses dados permitiram construir a curva de aprendizado do baseline tabular e observar se houve melhora progressiva no desempenho, plotada junto da do Deep Q Learning posteriormente.


A curva mostra uma melhora gradual ao longo do treinamento. O agente parte de uma situação em que praticamente todos os episódios terminam no pior caso possível e, com o passar dos episódios, passa a alcançar rewards menos negativos de forma recorrente. Isso confirma que a política aprendida conseguiu capturar a lógica de ganhar embalo e subir a montanha com mais eficiência.

### 4. Armazene os pesos da Q-table.

Ao final do treinamento, a **Q-table 3D** foi salva em arquivo. Como a tabela do `MountainCar` possui três dimensões, foi utilizado `pickle`, e não `np.savetxt`.

Arquivo salvo:

```text
data/q_table_mountaincar.pkl
```

Esse salvamento permite:

- reutilizar a política aprendida posteriormente;
- testar o agente sem treinar novamente;
- manter o baseline tabular persistido para comparação com o DQN.

### 5. Implemente o Deep Q-Learning para o ambiente MountainCar.

Nesta etapa, foi implementado o **Deep Q-Learning** para o `MountainCar-v0`. Em vez de uma Q-table, a função `Q(s,a)` passou a ser aproximada por uma rede neural.

A lógica do algoritmo foi:

- usar uma rede neural no lugar da Q-table;
- receber como entrada o estado contínuo `[posição, velocidade]`;
- produzir como saída `3` valores, um para cada ação;
- selecionar ações com política `epsilon-greedy`;
- armazenar experiências em memória;
- treinar a rede com *experience replay*;
- usar uma rede-alvo para dar mais estabilidade ao processo de treinamento.

#### Arquitetura da rede

A rede utilizada na configuração final foi:

- **camada oculta 1:** `128` neurônios;
- **camada oculta 2:** `128` neurônios;
- **camada de saída:** `3` neurônios.

No DQN, a rede neural substitui a tabela e tenta aprender padrões gerais entre estado e qualidade das ações. Assim, em vez de memorizar apenas estados discretos já visitados, ela passa a generalizar melhor para estados semelhantes.

### 6. Execute o treinamento n vezes onde este n precisa ser maior ou igual a 5.

Assim como no **Q-Learning tabular**, o treinamento do **Deep Q-Learning** foi repetido **5 vezes**, atendendo ao requisito do enunciado.

A melhor configuração encontrada foi então treinada nessas múltiplas execuções, e os rewards por episódio foram armazenados para análise posterior.

Os resultados foram:

- **Número de runs:** `5`
- **Melhor run escolhida:** `1`
- **Reward médio nos 50 primeiros episódios (média entre runs):** `-200.0`
- **Reward médio nos 50 últimos episódios (média entre runs):** `-118.548`

A diferença em relação ao baseline tabular foi tremenda, mesmo com um número ABSURDAMENTE menor de episódios no treinamento (15000 no Q-learning, 1000 no Deep Q). O agente começa, assim como o Q-Learning, preso no comportamento ruim de `-200`, mas termina com uma média final muito melhor.

### 7. Encontre os melhores hiperparâmetros para o Deep Q-Learning.

Foram testadas **cinco configurações diferentes** de hiperparâmetros do **Deep Q-Learning**. Cada uma representou um estilo de aprendizado distinto:

- **Base**;
- **Exploração mais longa**;
- **Menos foco no futuro**;
- **Batch menor**;
- **Target update mais frequente**.

A melhor configuração encontrada foi:

```python
label = "Target update mais frequente"
gamma = 0.99
epsilon_init = 1.0
epsilon_min = 0.01
epsilon_dec = 0.995
batch_size = 64
memory_size = 10000
learning_rate = 0.001
hidden_1 = 128
hidden_2 = 128
target_update_freq = 5
```

Resultado associado:

- **Reward médio nos 50 últimos episódios:** `-154.16`

Essa configuração foi a mais eficiente entre as testadas. O resultado sugere que atualizar a rede-alvo com mais frequência ajudou a estabilizar o processo de aprendizado e levou a uma evolução mais clara do agente ao longo do treinamento.

### 8. Colete todos os dados para criar a curva de aprendizado.

Assim como no baseline tabular, os rewards por episódio do **DQN** foram armazenados ao longo do treinamento. Isso permitiu construir:

- curvas médias por episódio;
- curvas suavizadas;
- comparação entre as configurações testadas;
- comparação final com o Q-Learning tabular.

**![Curva de aprendizado das configurações do DQN](Grafico5QSuav.png)**

A comparação entre as cinco curvas mostrou com clareza que a configuração **Target update mais frequente** foi a que melhor conseguiu sair do platô inicial de `-200` e atingir rewards médios menos negativos. As demais configurações também apresentaram alguma evolução, mas em intensidade claramente inferior.

### 9. Armazene os pesos da rede neural.

Ao final da busca e do treinamento final, foram salvos os pesos da melhor run da melhor configuração do **Deep Q-Learning** em arquivo, permitindo posterior reutilização do modelo treinado.

Arquivo salvo:

```text
data/dqn_mountaincar_best.pth
```

### 10. Compare os resultados obtidos com o Q-Learning e o Deep Q-Learning usando um plot. Tente deixar claro no plot qual a meta de recompensa acumulada para o ambiente MountainCar.

Com ambos os agentes treinados, foi construído um gráfico comparando suas curvas de aprendizado. Nesse vemos que:

- o **Deep Q-Learning** aparece com a melhor configuração encontrada;
- a linha de referência em `-200` indica um episódio sem sucesso, já que no `MountainCar-v0` a recompensa costuma ser `-1` por passo.

Como valores menos negativos são melhores, curvas que sobem acima de `-200` indicam aprendizado mais eficiente.

**![Comparação entre Q-Learning e Deep Q-Learning](GraficoAprendQs.png)**

Os resultados observados foram:

- **Reward final médio do Q-Learning tabular:** `-171.865`
- **Reward final médio do Deep Q-Learning:** `-118.548`

O gráfico deixa claro que o **Deep Q-Learning** conseguiu aprender uma política muito melhor do que a abordagem tabular. Enquanto o Q-Learning melhora de forma consistente, mas permanece em uma faixa mais modesta de desempenho, o DQN alcança rewards médios muito mais altos **mesmo com apenas questão de 7% do número de episódios**.

### 11. Crie outro gráfico que mostra o desempenho de ambos os agentes durante o processo de inferência, ou seja, quando o agente está atuando sem treinamento.

Agora:

- o agente tabular escolhe sempre a ação de maior valor na Q-table;
- o DQN escolhe sempre a ação de maior valor previsto pela rede.


**![Comparação de métricas na inferência](GraficoInfQs.png)**

**![Reward por episódio de teste na inferência](GraficoRewardsQs.png)**

Os resultados numéricos foram:

#### Q-Learning tabular

- **Taxa de sucesso:** `87.00%`
- **Reward médio:** `-138.18`
- **Número médio de ações:** `138.18`
- **Número médio de ações nos sucessos:** `128.94`

#### Deep Q-Learning

- **Taxa de sucesso:** `100.00%`
- **Reward médio:** `-102.56`
- **Número médio de ações:** `102.56`
- **Número médio de ações nos sucessos:** `102.56`

Na inferência, a supremacia do **Deep Q-Learning** ficou ainda mais evidente, inegável. O agente baseado em rede neural atingiu `100%` de sucesso, além de apresentar reward médio melhor e menos passos por episódio. O agente tabular também teve desempenho bom, mas ainda mostrou episódios mais longos e uma taxa de sucesso inferior. Na prática, o DQN aprendeu uma política mais eficiente e mais estável.

---

## Conclusão

De forma geral, esta atividade permitiu comparar diretamente duas abordagens importantes de aprendizagem por reforço baseada em valor no `MountainCar-v0`.

O **Q-Learning tabular** funcionou como um baseline sólido e reaproveitou com sucesso a lógica desenvolvida na APS 4. A discretização do estado tornou o problema viável e o agente conseguiu sair do patamar inicial de episódios sempre ruins, chegando a uma média final de `-171.865` no treinamento e `87%` de sucesso na inferência.

O **Deep Q-Learning** mostrou-se claramente superior. A melhor configuração encontrada foi **Target update mais frequente**, e o treinamento final alcançou uma média de `-118.548` nos episódios finais. Na inferência, o DQN obteve *100%* de sucesso, reward médio de `-102.56` e episódios mais curtos, *com questão de 79% das ações* mostrando uma política mais eficiente e mais estável.

Em resumo, os resultados indicam que, para o `MountainCar-v0`, o **Deep Q-Learning** foi mais eficiente do que o **Q-Learning tabular**, tanto no treinamento quanto na atuação final sem aprendizado revelando um exemplo claro onde uma rede neural é muito preferível. Eu me pergunto quando não seria preferível, honestamente. Vale notar que tivemos que rodar Pytorch para não demorar tanto assim, e ainda assim demorou MILÊNIOS.
