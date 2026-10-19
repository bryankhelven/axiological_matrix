# Railway: primeiro deploy experimental do Axiological Studio

**Situação:** preparação técnica; só declarar "publicado" depois de obter uma URL real, conferir os logs e testar a interface nessa URL. A conta Railway precisa estar conectada/autorizada para que a coordenação faça a implantação.

## Projeto separado, sem domínio comprado

1. Railway > **New Project** > **Deploy from GitHub Repo**.
2. Escolha o repositório `bryankhelven/axiological_matrix`. Crie projeto isolado, nome sugerido `axiological-studio-preview`, para não interferir nas aplicações existentes.
3. No serviço, conecte a branch **`experiment/sentiment-contrast-eniac2026`**.
4. **Mantenha o diretório raiz do build na raiz do repositório**, pois o Dockerfile copia dados e resultados do projeto. Use Build > Dockerfile Path `studio/Dockerfile.core`, ou a variável de serviço `RAILWAY_DOCKERFILE_PATH=studio/Dockerfile.core`. Não use `studio/Dockerfile` para o primeiro teste: instala os pesos afetivos grandes.
5. Configure em **Variables**:
   - `AXIOLOGICAL_REQUIRE_AUTH=1`
   - `STUDIO_ACCESS_PASSWORD` = segredo único e longo (pelo menos 16 caracteres), gerado por você diretamente no Railway. **Não envie essa senha ao chat e não a comite no GitHub**.
   - `PYTHONUNBUFFERED=1` é opcional.
6. Deixe o `PORT` a cargo do Railway. O `studio/Dockerfile.core` já escuta no valor dinâmico `PORT`. Start Command: padrão do Dockerfile, sem sobrescrever.
7. Desative auto-deploy para esta branch durante a fase de pesquisa; implante manualmente um commit verificado. Verifique configurações de recursos e gastos, já que esse novo serviço acrescenta consumo à assinatura existente.
8. Faça o primeiro deploy sem domínio público. Verifique logs de instalação, treinamento e saúde. Caso tudo funcione, Settings > Networking > **Generate Domain** cria URL Railway com HTTPS, sem comprar domínio. Nunca exponha publicamente sem senha configurada e testada.
9. Abra o domínio gerado e confira: tela de senha aparece; senha inválida não dá acesso; senha correta libera texto novo; CSV próprio gera matrizes e exporta; R4 usa **somente** ZIP sanitizado. Teste o uso com navegador anônimo e verifique os recursos no dashboard.
10. Após o teste, se a aplicação for ficar pública, faça revisão de privacidade, direitos dos textos do Bluesky e limites de envio antes de disponibilizar a URL a terceiros.

## Evidência automática, já executada

- `.github/workflows/axiological-studio-ci.yml`: pacote dos modelos, critérios e API.
- `.github/workflows/axiological-studio-browser.yml`: Chromium real, login com senha **efêmera criada apenas na CI**, CSV, duas matrizes, pacote científico sanitizado de *fixture* e exportações.
- `.github/workflows/axiological-studio-container.yml`: contêiner leve iniciado na porta dinâmica e testado com healthcheck.

O pacote R4 para uso real existe fora do repositório por razões de proveniência/privacidade. **Nunca incluir arquivos privados de amostragem humana em deploy**. O bundle original não deve ser disponibilizado publicamente.

## Bloqueio de conta

O app Railway ainda não está conectado ao ChatGPT nesta conversa. A implantação numa conta real demanda que você autorize essa integração, ou realize os passos no dashboard Railway. Não afirmar que um serviço ou domínio foi criado até receber confirmação verificável da plataforma. Sem domínio comprado necessário.

## Problemas conhecidos

- O filtro TF-IDF pode dar falso positivo para texto completamente fora de domínio. Avisos de cobertura e uma triagem por margem estão implementados; não substituem o rótulo nem garantem acerto.
- O container leve **não executa** os modelos externos pesados de sentimentos e emoções. Há integração e smoke tests desses modelos em ambiente separado, mas recursos/custos de um serviço contínuo com todos os pesos ainda precisam de benchmark.
- A ferramenta foi validada como executável, não certificada quanto a qualidade de inferência, privacidade de produção ou resistência a ataques.
