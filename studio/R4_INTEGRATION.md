# Integração de resultados da frente Work (Rodada 4)

Estado: **código integrado e teste de esquema** na ramificação `experiment/sentiment-contrast-eniac2026`. A publicação de qualquer corpus adicional depende de verificar direitos e privacidade; nenhuma chave privada de avaliação humana pode ser incorporada.

## O que foi integrado

O Axiological Studio inclui a aba **Estudo Beyond Sentiment**, que recebe `Axiological_Studio_R4_Dados_Integracao_Seguros.zip`. O pacote seguro é derivado de `Beyond_Sentiment_Rodada4_Pacote_Experimental.zip` SHA-256 `e068b34051aec93f9de7b67d2b82bf18f14e54ae32d4c9703282d09916289137`.

A transformação:

- retém as 733 predições comparativas de quatro modelos de sentimento e dois de emoção e suas 28 probabilidades multirrótulo;
- mantém métricas de informação condicional, tabelas de validação cruzada de sondagens e frequências emocionais;
- omite o campo de texto integral e todos os arquivos de `review_preparation`, inclusive a chave privada de amostragem;
- exige checksums SHA-256 no manifesto e reconcilia `item_id` + `text_hash` e os quatro campos de anotação A1/A2 com o gold público no próprio repositório;
- não altera rótulos humanos originais e não inventa consenso onde há discordância.

## Usar

1. Abra o Axiological Studio a partir do repositório.
2. Na aba **Estudo Beyond Sentiment**, envie **somente o ZIP seguro** `Axiological_Studio_R4_Dados_Integracao_Seguros.zip`, preparado pela conversa de coordenação.
3. Escolha o anotador original, o modelo de sentimento, o grupo de discordância, o rótulo emocional e o limiar.
4. Leia os cruzamentos e exporte um recorte. Tabelas de informação condicional e validação cruzada ficam em uma seção separada e não são relatadas como desempenho do Studio.
5. Para analisar **novos textos seus**, use as abas **Analisar publicação** ou **Analisar CSV**; o experimento Work é um conjunto histórico suplementar separado dessas funções.

## Limites

- O pacote científico original contém chaves e instruções de revisão futura. **Jamais enviar o ZIP original para uma implantação pública.**
- A aba não treina novos modelos nem reavalia o gold original.
- A validação cruzada de intensidade presente no pacote Work usa orientação humana como atributo explicativo e **não é desempenho de inferência axiológica do Studio**.
- Sentimentos e emoções são predições externas. Não há gold novo de emoções/sentimentos no Bluesky e suas probabilidades não são equivalentes a certeza.
- A anotação humana de item 1368 permanece inalterada.
- `paired_axis_annotations_733.csv` é a fonte textual local para recuperar os textos, com verificação cruzada das chaves. Publicação em app aberto depende de decisão sobre licenciamento dos textos.
- Esta integração não representa implantação pública e não implica sincronização automática entre as duas conversas.

## Testes

`python studio/test_work_r4.py` verifica esquema, checksums, cobertura dos 733, rótulos humanos preservados, quatro grupos de concordância (395/176/67/95), rejeição de arquivo privado, rejeição de hash trocado e rejeição de manifesto incompatível.

Esse teste usa predições **sintéticas apenas na fixture de teste**; não representa outro experimento científico. As saídas da Rodada 4 foram produzidas exclusivamente pela frente Work e preservadas na entrega original.
