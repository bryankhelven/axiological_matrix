# Coordenação das frentes: Axiological Matrix / Beyond Sentiment

Data de referência: 19/10/2026. **Contrato de trabalho e integração, não confirmação de resultados futuros.**

## 1. Regra geral

- **Work / frente científica** responde por evidência nova, comparações, avaliação, documentação científica e pacotes reproduzíveis.
- **Conversa principal / integração** responde por Axiological Studio, classificadores axiológicos, inferência de textos novos, testes de produto, API, CSV, implantação, seleção de evidência para comunicação e gestão das entregas.
- **Sem sincronização automática entre conversas.** Ao concluir um marco, Work entrega ZIP + relatório de handoff ao usuário; este os encaminha à conversa principal. Esta conversa confere, aceita ou devolve com pendências.
- Todas as alterações remotas da coordenação ficam na branch `experiment/sentiment-contrast-eniac2026`. Não alterar `main`, camera-ready ou gold humano. Work não faz commits na branch compartilhada, a menos que haja autorização expressa e divisão de arquivos acordada.

## 2. Fonte canônica para evitar duplicação

Entrada para Work: `HANDOFF_COMPLETO_FRENTE_CIENTIFICA_BEYOND_SENTIMENT_20261019(1).zip`, contendo o pacote experimental da rodada 4 e o handoff de 19/10. A rodada 4 já reporta **4 modelos sentimentais, 2 emocionais**, uma base comum de **733 pares**, **3 modelos novos executados nos 1.497 textos**, análises condicionais da intensidade e 15 leituras de exemplos. Não reexecutar automaticamente esses experimentos.

Entrada adicional disponível na branch experimental: `experiments/axis_disagreement/audit_existing.py` e auditoria dos 733 pares (395 concordantes nos dois eixos; 176 só discordância de intensidade; 67 só de aceitação; 95 discordância em ambos). É um insumo já reproduzido, não uma nova avaliação humana.

O Work não deve repetir o desenvolvimento da UI/API/Streamlit, exportação dos cinco TF-IDF axiológicos ou experimentos de classificação de texto novo no Studio. A coordenação não deve reproduzir seus experimentos de alvo/stance ou benchmark externo emocional sem necessidade de auditoria de integração.

## 3. Escopo e ordem da frente científica (Work)

### W1. Comparador orientado ao alvo e/ou stance

Testar uma comparação cientificamente justa de sentimento global versus avaliação do **alvo IA e educação** ou posicionamento quanto a esse objeto, distinguindo alvo, posição e intensidade. O objeto inclui ensino **sobre** IA, usos, crítica, formação, discurso e educação, e não apenas adoção de IA na sala de aula.

**Entrega mínima de aceite:** inventário justificado de modelos e suporte ao português; revisão/commit do modelo e licença; protocolo ou template congelado ANTES da inferência; universos e partições fixados; CSV por publicação com `item_id`, `text_hash`, predição, escores, modelo/revisão e campos de incerteza; comparação quantitativa e análise de alvos; limites e contraexemplos; scripts e requisitos reproduzíveis. Não usar aceitação como se fosse gold sentimental.

### W2. Benchmark emocional externo

Usar corpus externo inspecionável (BRIGHTER, se compatível, ou alternativa justificável) com gold emocional real. Checar português, etiquetas, partições, licença, origem/traduções, interseção com o treinamento dos checkpoints. Preservar inventários diferentes (quatro emoções XLM-EMO versus 28 escores multirrótulo do modelo português). Definir limiar usando somente treino/validação, nunca teste.

**Entrega mínima de aceite:** fonte, licença, split imutável, código, arquivos de predição, métricas pertinentes a multirrótulo (micro/macro-F1 e métricas por rótulo quando aplicáveis), incerteza, análise de erros e limites. Resultados externos NÃO são acurácia no Bluesky.

### W3. Robustez e intensidade com desacordo

Usar A1 e A2, **733 pares válidos** (734 aplicáveis; item 298 sem anotação A1 dos eixos), 395 consensuais em 3 classes e 244 em 5 classes. Investigar distância ordinal, confiabilidade de intensidade, modelos consenso/soft-label, OOF, calibração, reamostragem por `text_hash`, sensibilidade a anotadores e fontes afetivas, inclusive modelos de alvo. Comparar regimes com as mesmas divisões. Distinguir objetivo de probabilidades adequadas de objetivo de classe rígida. Testar hipóteses que podem REFUTAR benefício incremental.

**Entrega mínima de aceite:** tabelas por subgrupo de concordância e discordância, resultados probabilísticos e discretos lado a lado, intervalos e testes quando defensáveis; dois ou mais contraexemplos; critérios formais e explicação de limites. O fato de 265/271 divergências de intensidade de 3 classes serem adjacentes não comprova que os rótulos estejam corretos.

### W4. Análise discursiva sistemática

Definir protocolo operacional de alvo, voz citada, avaliação múltipla, concessão, ironia, negação, força avaliativa, enunciados difíceis e controles. Anotar **apenas como leitura analítica do pesquisador/assistente**, sem transformar em gold ou inventar concordância humana. Não revisar ou alterar rótulos humanos originais. Em particular, **não reabrir a etiqueta original de item 1368**; se um modelo divergir, registrar como contraste sem declarar erro do gold.

**Entrega mínima de aceite:** exemplos reais com `item_id`, `text_hash`, fragmento fiel, categorias operacionais, explicação, contraexemplos e ligação verificável às contagens.

### W5. Consolidação e entrega de evidência

Entregar um pacote de integração, não slides nem aplicação. Priorizar W1 e W2 antes de expandir W3/W4. Após cada marco, enviar material incremental, sem esperar que todos os marcos terminem.

## 4. Escopo da coordenação / Studio (esta conversa)

**P1:** manter infraestrutura da branch experimental, pipelines TF-IDF versionados, API, Streamlit, entrada de texto, upload CSV, download CSV, matriz do corpus do usuário e transparência de gate; testar erros e integridade ponta a ponta.

**P2:** realizar testes técnicos de robustez e recursos, segurança mínima e documentação; distinguir benchmark em domínio original, teste externo, inferência em texto novo e avaliação de produto. O `studio/audit_applicability.py` avalia 50% por grupos de hashes; rótulos de revisão por margem não substituem o gate.

**P3:** integrar os resultados W1-W4 após validação de manifesto e chaves, com nomes de modelos/revisões, versão de dataset, avisos, escores e limiares. Não expor ao público dados privados de cegamento e materiais não licenciados.

**P4:** testar imagem/aplicação Docker completa, integrar perfis de implantação autorizados e obter URL pública apenas se a implantação for efetivamente verificada. Sem prometer que há app publicado antes disso.

**P5:** selecionar figuras, métricas e exemplos auditados para uma apresentação revisada quando houver resultados maduros. Não editar o artigo original sem novo pedido.

## 5. Contrato técnico entre Work e Studio

Cada pacote de Work deve conter:

`HANDOFF.md`: data, identificador da rodada, marcos concluídos/abertos, perguntas e limitações, o que mudou.

`manifest.json`: caminhos relativos, contagens, universos, `sha256`, revisões dos modelos, versões do ambiente, critérios de exclusão, identificadores de splits, licença e status `confirmed/exploratory/pending`.

`predictions/*.csv`: `item_id`, `text_hash`, `source_universe`, `model_id`, `model_revision`, `task`, `label_inventory`, `prediction`, escores por classe ou rótulo; campos de truncamento. Para emoção multirrótulo, manter **todos** os escores e limiar registrado, não forçar argmax.

`metrics/*.csv` e `figures/*`: métricas com denominador e rótulo da tarefa, e figuras geradas pelos scripts.

`scripts/*`, `environment/*`: repro do pacote, sem dependência de caminhos temporários de uma sessão.

`INTEGRATION.md`: quais novas abas/gráficos/resultados o Studio deve mostrar e qual saída é só exploratória.

`PUBLICATION_SAFETY.md`: quais campos podem ser divulgados; não encaminhar a chave de estratificação de anotação aos anotadores e não incluir em build público.

**Critérios de aceite desta conversa:**
1. Integridade ZIP e SHA-256; manifesto consistente.
2. Alinhamento `item_id` **e** `text_hash`, nunca junção posicional.
3. Números sempre com denominador (1.497, 734, 733, 395 ou universo externo).
4. Preservação de A1/A2, sem consenso sintético.
5. Protocolo sem vazamento de validação/teste e sem atribuir causalidade ou representatividade indevidas.
6. Teste de reconstrução das tabelas/figuras a partir das predições, com tolerâncias justificadas.
7. Integração aprovada no Studio e documentação atualizada. Se falhar, devolver checklist de inconsistências ao Work.

## 6. Regras transversais

- Nenhuma nova rodada humana antes de todas as etapas computacionais e de produto. O Work pode preparar formulários **vazios**, mas não solicitar preenchimento.
- Corpus de **IA e educação** no sentido amplo. Não reduzir ao apoio à adoção pedagógica.
- Aceitação e intensidade são dimensões **conceitualmente distintas**, não presumidamente estatisticamente independentes.
- Aplicabilidade é um **gate** anterior, não terceiro eixo.
- Sentimento, emoção, alvo/stance e orientação axiológica não são classes intercambiáveis.
- O artigo aceito é congelado; resultados posteriores são suplementares, visivelmente identificados.
- Não escolher modelos por maximizar divergência com a matriz.
- Preservar a história de execuções: novo modelo/revisão gera nova versão, não sobrescreve predições antigas.
- Interface pública não revela tokens de acesso, IDs de anotação internos, chaves privadas, ou conteúdo sem autorização.

## 7. Ordem de comunicação

1. A coordenação entrega este contrato ao usuário para encaminhar ao Work.
2. Work confirma recebimento, objetivos W1/W2 e informa fontes e plano mínimo antes de rodar modelos.
3. Work entrega um ZIP incremental por marco, com manifesto e relatório.
4. O usuário anexa o ZIP nesta conversa.
5. A coordenação audita e integra, ou devolve falhas concretas.
6. Work retoma W3/W4 e entrega próximos marcos; produto avança em paralelo.
7. **Somente após fechamento experimental e do produto**, decidir e executar eventualmente as anotações humanas.

Última observação: este contrato não implica execução em segundo plano ou transmissão automática entre conversas. 
