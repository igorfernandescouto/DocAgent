# DocAgent

Projeto simples e funcional para demonstrar uma API com:

- RAG para responder perguntas com base em documentos
- Tools para salvar informacoes e chamar uma API externa
- FastAPI como camada de exposicao
- Logs estruturados
- Avaliacao basica de latencia, grounding e cache
- Estrutura pronta para rodar localmente ou via Docker

O foco aqui e ser facil de explicar em entrevista, com codigo limpo, responsabilidades bem separadas e pouca logica aninhada.

## O que este projeto demonstra

- Agente de IA: orquestracao entre recuperacao de contexto, decisao e tools
- RAG: ingestao, chunking, embeddings, ranking e resposta condicionada ao contexto
- Integracao: API HTTP e chamada externa opcional
- Observabilidade: logs estruturados e metricas simples
- Avaliacao: latencia, tokens estimados e groundedness
- Deploy: Dockerfile e `render.yaml`

## Arquitetura

Fluxo principal:

1. O usuario envia uma pergunta para `/ask`
2. O sistema recupera chunks relevantes do indice vetorial
3. O agente decide se deve responder, resumir ou executar uma tool
4. A resposta final e devolvida com fontes e metricas
5. A execucao e registrada em log estruturado

Camadas:

- `app/main.py`: API e endpoints
- `app/container.py`: composicao das dependencias
- `app/services/rag_service.py`: ingestao e recuperacao
- `app/services/agent_service.py`: orquestracao principal
- `app/services/tool_service.py`: automacoes simples
- `app/core/`: chunking, avaliacao, logging e utilitarios

## Decisoes de projeto

- Embeddings locais por hash: deixam o projeto funcional mesmo sem chave de API.
- OpenAI compativel opcional: se `OPENAI_API_KEY` e os modelos forem configurados, o projeto usa chamadas remotas para embeddings e geracao.
- Vector store persistido em JSON: mantem a execucao simples e facil de inspecionar. A interface foi isolada para facilitar troca futura por FAISS ou Chroma.
- Fallback claro: se nao houver contexto suficiente, o agente responde que nao encontrou base suficiente em vez de inventar.
- Cache em memoria: acelera perguntas repetidas.

## Estrutura

```text
app/
  core/
  services/
  config.py
  container.py
  main.py
data/
  index/
  logs/
  sample_docs/
  tool_storage/
scripts/
  run_eval.py
Dockerfile
render.yaml
requirements.txt
```

## Como rodar localmente

### 1. Criar ambiente e instalar dependencias

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configurar ambiente

```powershell
Copy-Item .env.example .env
```

Se voce quiser usar um modelo remoto, preencha:

- `OPENAI_API_KEY`
- `OPENAI_CHAT_MODEL`
- `OPENAI_EMBEDDING_MODEL`

Se deixar em branco, o projeto continua funcional com fallback local.

### 3. Subir a API

```powershell
uvicorn app.main:create_app --factory --reload
```

A API fica em `http://127.0.0.1:8000`.

## Endpoints

### `GET /health`

Retorna o estado do servico, quantidade de documentos e modo de embeddings.

### `GET /documents`

Lista os documentos atualmente indexados.

### `POST /ingest`

Aceita upload de arquivo (`.txt`, `.md`, `.pdf`) ou texto manual.

Exemplo com texto:

```powershell
curl -X POST "http://127.0.0.1:8000/ingest" `
  -F "source_name=politica_customizada.txt" `
  -F "text=Clientes enterprise possuem SLA de 4 horas em horario comercial."
```

Exemplo com arquivo:

```powershell
curl -X POST "http://127.0.0.1:8000/ingest" `
  -F "file=@data/sample_docs/politica_devolucao.txt"
```

### `POST /ask`

Pergunta algo para o agente.

```json
{
  "question": "Qual e a politica de devolucao?"
}
```

Resposta esperada:

- `answer`: resposta final
- `action`: `answer`, `summarize` ou `tool`
- `sources`: chunks usados
- `metrics`: latencia, groundedness, cache hit e tokens estimados

### `GET /logs`

Retorna as ultimas execucoes com pergunta, resposta, fontes, tool e metricas.

### `GET /metrics`

Retorna um resumo agregado das execucoes recentes.

## Tools implementadas

### `save_candidate_interest`

Salva um lead interessado em `data/tool_storage/interested_leads.json`.

Exemplo:

```json
{
  "question": "Salve esse cliente como interessado. Nome: Ana Silva Email: ana@empresa.com"
}
```

### `call_external_api`

Envia um payload para `EXTERNAL_API_BASE_URL/leads`.

Se a variavel nao estiver configurada, a chamada e simulada e registrada em `data/tool_storage/external_api_calls.jsonl`.

Exemplo:

```json
{
  "question": "Envie este lead para a API externa com email joao@empresa.com"
}
```

## Avaliacao

Rodar avaliacao simples:

```powershell
python scripts/run_eval.py
```

O script mede:

- latencia media
- groundedness medio
- se a resposta contem palavras-chave esperadas
- se a acao escolhida bate com o esperado

## Exemplos de perguntas para demonstracao

- `Qual e a politica de devolucao para um produto sem defeito?`
- `Resuma o documento do Plano Premium`
- `Salve esse cliente como interessado. Nome: Carla Email: carla@empresa.com`
- `Envie esse lead para a API externa com email carla@empresa.com`

## Como explicar na entrevista

Uma forma simples de apresentar:

1. "Eu separei o projeto em ingestao, recuperacao, decisao e execucao."
2. "O RAG recupera contexto relevante e a resposta e limitada a esse contexto."
3. "O agente consegue tanto responder quanto executar automacoes simples."
4. "Cada execucao gera logs e metricas para observabilidade e avaliacao."
5. "Mantive a infraestrutura leve, mas deixei os contratos prontos para trocar o vector store por FAISS/Chroma e para plugar um modelo remoto."

## Deploy

### Docker

```powershell
docker build -t docagent .
docker run -p 8000:8000 --env-file .env docagent
```

### Render

O arquivo `render.yaml` ja descreve um deploy simples via Docker.

## Melhorias naturais para a proxima iteracao

- Trocar o vector store local por FAISS ou Chroma
- Persistir cache em Redis
- Adicionar autenticacao na API
- Criar suite de testes automatizados
- Adicionar dataset de avaliacao maior
