from crewai import Agent

# from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools
from tools.sec_tools import SECTools
from textwrap import dedent
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.chat_models import ChatOpenAI



class StockAnalysisAgents():
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        # self.Ollama = Ollama(model="openhermes")

    def financial_analyst(self):
        return Agent(
            role="The Best Financial Analyst",
            backstory=dedent("""The most seasoned financial analyst with lots of expertise
                        in short-term stock market analysis and investment strategies.
                        Working for a super important customer, your goal is to accurately
                        predict tomorrow's stock prices based on the latest market trends and data."""),
            goal=dedent("""Impress all customers by providing accurate predictions for tomorrow's stock prices
                    through comprehensive short-term financial data analysis."""),
            tools=[SearchTools.search_internet, YahooFinanceNewsTool(), CalculatorTools.calculate, SECTools.search_10k, SECTools.search_10q],
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def research_analyst(self):
        return Agent(
            role="Staff Research Analyst",
            backstory=dedent("""Known as the BEST research analyst, you're skilled in sifting through news,
                        company announcements, and market sentiments to provide short-term analysis. 
                        Now you're working on a super important customer. Your goal is to accurately
                        predict tomorrow's stock prices by interpreting the latest market information."""),
            goal=dedent("""Be the best at gathering and interpreting data to provide accurate predictions
                        for tomorrow's stock prices based on the latest market information."""),
            tools=[SearchTools.search_internet, YahooFinanceNewsTool(), CalculatorTools.calculate, SECTools.search_10k, SECTools.search_10q],
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def intelligent_investment_advisor(self):
        return Agent(
            role="Intelligent Investment Advisor",
            backstory=dedent("""Renowned for your exceptional analytical skills and foresight, you excel in predicting market trends and guiding clients towards profitable investments. With a keen eye for detail, your goal is to provide not just insights but actionable recommendations, including precise classification of stocks based on future price movements."""),
            goal=dedent("""Your objective is to deliver accurate analyses and predictions for tomorrow's stock prices, leveraging advanced classification techniques to categorize stocks into appropriate risk categories. Through your expertise, you aim to empower clients to make informed investment decisions with confidence."""),
            tools=[SearchTools.search_internet, YahooFinanceNewsTool(), CalculatorTools.calculate],
            verbose=True,
            llm=self.OpenAIGPT35
        )


