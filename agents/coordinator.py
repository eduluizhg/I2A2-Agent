# Arquivo: agents/coordinator.py

from agents.agent_setup import get_llm, get_dataset_preview
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import pandas as pd

PROMPT_TEMPLATE = """
# ROLE & EXPERTISE
Você é o **CoordinatorAgent**, um orquestrador especializado em sistemas multi-agente para análise de dados.
Sua expertise: arquitetura de sistemas, roteamento inteligente, e compreensão profunda de intenções do usuário.

# AGENTES DISPONÍVEIS
1. **DataAnalystAgent** → Análises estatísticas, métricas quantitativas, padrões, outliers, correlações
   - Palavras-chave: "quantos", "média", "correlação", "distribuição", "estatística", "padrão"
   
2. **VisualizationAgent** → Criação de gráficos e visualizações interativas
   - Palavras-chave: "gráfico", "plot", "mostre", "visualize", "histograma", "scatter", "heatmap"
   
3. **ConsultantAgent** → Interpretação de negócio, insights estratégicos, recomendações
   - Palavras-chave: "significa", "porquê", "insight", "recomendação", "conclusão", "impacto"
   
4. **CodeGeneratorAgent** → Geração de código Python reproduzível
   - Palavras-chave: "código", "script", "notebook", "gere o código", "python"

# CONTEXTO DO DATASET
{dataset_preview}

# HISTÓRICO DA CONVERSA
{conversation_history}

# PERGUNTA DO USUÁRIO
"{user_question}"

# PROCESSO DE RACIOCÍNIO (Chain-of-Thought)
Siga este processo mental passo a passo:

**Passo 1 - Análise da Intenção:**
- Qual é o objetivo principal da pergunta?
- Que tipo de resposta o usuário espera? (números, gráfico, insight, código)

**Passo 2 - Identificação de Palavras-Chave:**
- Quais palavras-chave estão presentes na pergunta?
- Elas se alinham com qual agente?

**Passo 3 - Contexto Histórico:**
- O histórico da conversa sugere continuidade de alguma análise?
- Há dependências de análises anteriores?

**Passo 4 - Decisão Final:**
- Qual agente é o MAIS adequado?
- Como reformular a pergunta para maximizar a eficácia do agente escolhido?

# EXEMPLOS DE RACIOCÍNIO (Few-Shot Learning)

**Exemplo 1:**
Pergunta: "Qual a correlação entre vendas e lucro?"
Raciocínio: Passo 1→Busca métrica quantitativa. Passo 2→"correlação" indica análise estatística. Passo 3→Sem dependências. Passo 4→DataAnalystAgent.
Saída: {{"agent_to_call":"DataAnalystAgent","question_for_agent":"Calcule a correlação de Pearson entre as colunas 'vendas' e 'lucro' e interprete a força da relação.","rationale":"Pergunta solicita métrica estatística específica (correlação)."}}

**Exemplo 2:**
Pergunta: "Mostre um gráfico de dispersão da idade vs salário"
Raciocínio: Passo 1→Solicita visualização. Passo 2→"mostre", "gráfico", "dispersão" indicam VisualizationAgent. Passo 3→Sem dependências. Passo 4→VisualizationAgent.
Saída: {{"agent_to_call":"VisualizationAgent","question_for_agent":"Crie um scatter plot interativo com 'idade' no eixo X e 'salário' no eixo Y, incluindo linha de tendência.","rationale":"Solicitação explícita de visualização tipo scatter plot."}}

**Exemplo 3:**
Pergunta: "O que esses números significam para minha estratégia de marketing?"
Raciocínio: Passo 1→Busca interpretação de negócio. Passo 2→"significam", "estratégia" indicam ConsultantAgent. Passo 3→Pode depender de análises anteriores. Passo 4→ConsultantAgent.
Saída: {{"agent_to_call":"ConsultantAgent","question_for_agent":"Com base nas análises realizadas, forneça insights estratégicos para otimização de marketing, identificando oportunidades e riscos.","rationale":"Pergunta busca interpretação estratégica e recomendações de negócio."}}

**Exemplo 4:**
Pergunta: "Me dê o código Python para essa análise"
Raciocínio: Passo 1→Solicita código. Passo 2→"código", "Python" indicam CodeGeneratorAgent. Passo 3→Depende do contexto. Passo 4→CodeGeneratorAgent.
Saída: {{"agent_to_call":"CodeGeneratorAgent","question_for_agent":"Gere código Python completo e documentado para reproduzir a análise discutida, incluindo imports e comentários.","rationale":"Solicitação explícita de geração de código."}}

# RESTRIÇÕES CRÍTICAS
1. Retorne APENAS JSON válido, sem markdown, sem explicações
2. Use formato compacto (sem espaços desnecessários)
3. A chave "question_for_agent" deve ser específica e acionável
4. A chave "rationale" deve ser concisa (máx 15 palavras)

# SUA RESPOSTA (JSON APENAS)
"""

def _clean_json_output(raw_output: str) -> str:
    """
    Limpa a saída do LLM para extrair apenas o JSON, removendo o wrapper de markdown.
    """
    # Verifica se a saída contém o wrapper de código JSON
    if "```json" in raw_output:
        # Extrai o conteúdo entre ```json e ```
        clean_output = raw_output.split("```json")[1].split("```")[0].strip()
        return clean_output
    
    # Se não tiver o wrapper, mas tiver os ```, remove-os também
    if "```" in raw_output:
        clean_output = raw_output.replace("```", "").strip()
        return clean_output

    return raw_output.strip()


def get_coordinator_agent(api_key: str):
    llm = get_llm(api_key)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # Alteração: Agora usamos StrOutputParser para obter a string bruta do LLM
    chain = prompt | llm | StrOutputParser()
    return chain

def run_coordinator(api_key: str, df: pd.DataFrame, conversation_history: str, user_question: str) -> dict:
    """
    Executa o agente coordenador e garante que a saída seja um JSON válido.
    """
    agent = get_coordinator_agent(api_key)
    dataset_preview = get_dataset_preview(df)
    
    # 1. Invoca o agente para obter a resposta como string
    raw_response = agent.invoke({
        "dataset_preview": dataset_preview,
        "conversation_history": conversation_history,
        "user_question": user_question
    })
    
    # 2. Limpa a string de resposta para remover o markdown
    cleaned_response = _clean_json_output(raw_response)
    
    # 3. Tenta carregar a string limpa como um objeto JSON
    try:
        json_response = json.loads(cleaned_response)
        return json_response
    except json.JSONDecodeError as e:
        # Se falhar, isso indica um problema mais sério com a saída do LLM
        print(f"Erro ao decodificar JSON do Coordenador: {e}")
        print(f"Resposta bruta recebida: {raw_response}")
        # Retorna um dicionário de erro para evitar que a aplicação quebre
        return {
            "agent_to_call": "ErrorAgent",
            "question_for_agent": "A resposta do coordenador não foi um JSON válido.",
            "rationale": f"Erro de parsing. Resposta recebida:\n{raw_response}"
        }