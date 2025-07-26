#generator.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

load_dotenv()

def generate_answer(query, retrieved_chunks, chat_history=None):
    # Create LangChain chat model with OpenRouter
    llm = ChatOpenAI(

        model="anthropic/claude-3-haiku:beta",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3
      )
    
    # System prompt
    BANGLA_SYSTEM_PROMPT = """
নিম্নলিখিত নির্দেশাবলী অনুসরণ করুন:
১. শুধুমাত্র প্রদত্ত প্রসঙ্গ ব্যবহার করুন
২. সংখ্যা ও নাম হুবহু উল্লেখ করুন
৩. উত্তর প্রশ্নের ভাষায় দিন
৪. অপ্রাসঙ্গিক বিশ্লেষণ এড়িয়ে চলুন

উত্তরের ফরম্যাট:
উত্তর: [স্পষ্ট উত্তর]
সূত্র: [প্রাসঙ্গিক উদ্ধৃতি]
"""

    system_prompt = "Answer the following question using the given context using exact value. Use specific keywords. Support Bangla and English. Answer in the same language as the question"

    lang = "bangla" if any("\u0980" <= char <= "\u09FF" for char in query) else "english"

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        #SystemMessagePromptTemplate.from_template(system_prompt),
        SystemMessagePromptTemplate.from_template(BANGLA_SYSTEM_PROMPT if lang == "bangla" else system_prompt),
        *([] if not chat_history else chat_history),
        HumanMessagePromptTemplate.from_template(
            "Context:\n{context}\n\nQuestion:\n{query}"
        )
    ])
    
    # Format the prompt
    formatted_prompt = prompt.format_messages(
        context="\n".join(retrieved_chunks),
        query=query
    )
    
    try:
        response = llm.invoke(formatted_prompt)
        return response.content.strip()
    except Exception as e:
        return f"❌ Error: {e}"
    
