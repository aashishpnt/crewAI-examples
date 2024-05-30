from crewai import Crew
from textwrap import dedent
import pandas as pd

from stock_analysis_agents import StockAnalysisAgents
from stock_analysis_tasks import StockAnalysisTasks

from dotenv import load_dotenv
load_dotenv()

class FinancialCrew:
    def __init__(self, companies):
        self.companies = companies

    def run(self):
        results = []

        for company in self.companies:
            agents = StockAnalysisAgents()  # Create new agents for each company
            tasks = StockAnalysisTasks(company)  # Create new tasks for each company

            research_analyst_agent = agents.research_analyst()
            financial_analyst_agent = agents.financial_analyst()
            investment_advisor_agent = agents.intelligent_investment_advisor()

            research_task = tasks.research(research_analyst_agent, company)
            financial_task = tasks.financial_analysis(financial_analyst_agent)
            filings_task = tasks.filings_analysis(financial_analyst_agent)
            classification_task = tasks.classify_stock(investment_advisor_agent)

            crew = Crew(
                agents=[research_analyst_agent, financial_analyst_agent, investment_advisor_agent],
                tasks=[research_task, financial_task, filings_task, classification_task],
                verbose=True
            )   
            result = crew.kickoff()
            results.append(result)
        return results

if __name__ == "__main__":
    print("## Welcome to Financial Analysis Crew")
    print('-------------------------------')
    
    def get_stock_to_analyze():
        ### filtering top 5 stocks from positively and negatively predicted classes
        df = pd.read_csv('predictions_with_true.csv')
        df = df[df['date'] == df['date'].max()]
        top_5_elements_class_0 = df.nlargest(5, 'prob_class_0')
        top_5_elements_class_2 = df.nlargest(5, 'prob_class_2')
        limited_df = pd.concat([top_5_elements_class_0, top_5_elements_class_2])
        return list(limited_df['ticker'])
    
    companies = get_stock_to_analyze()
    
    
    financial_crew = FinancialCrew(companies)
    results = financial_crew.run()
    print("\n\n########################")
    print("## Here is the Report")
    print("########################\n")
    for result in results:
        print(result)
    def save_to_markdown(filename, data):
        with open(filename, 'w') as f:
            f.write(data)
    save_to_markdown('result_merged.md', '\n'.join(results))