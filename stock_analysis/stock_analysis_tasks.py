from crewai import Task
from textwrap import dedent
import logging
from datetime import datetime, timedelta

logging.basicConfig(filename='crew_ai_logs.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class StockAnalysisTasks():
  def __init__(self, company):
      self.company = company
      
  def research(self, agent, company):
      two_days_ago = datetime.now() - timedelta(days=2)
      formatted_date = two_days_ago.strftime("%Y-%m-%d")

      return Task(description=dedent(f"""
          Collect and summarize news articles, press releases, and market analyses related to {company}
          and its industry published within the last two days (since {formatted_date}).
          Pay special attention to significant events, market sentiments, and analysts' opinions.
          Include upcoming events like earnings and others.

          Your final answer MUST be a report that includes a comprehensive summary of the latest news,
          any notable shifts in market sentiment, and potential impacts on the stock.
          Also, make sure to return the stock ticker: {company}.

          {self.__tip_section()}

          Make sure to use the most recent data as possible.

          Selected company by the customer: {company}
          """),
          agent=agent
      )
  
  def financial_analysis(self, agent):
    logging.info(f"{agent} is performing financial analysis task")
    return Task(
        description=dedent(f"""
            Conduct a thorough analysis of the financial health and market performance of the stock symbol: {self.company}.
            Examine key financial metrics such as P/E ratio, EPS growth, revenue trends, and debt-to-equity ratio.
            Also, analyze the performance of {self.company} in comparison to its industry peers and overall market trends.
            
            Your final report MUST expand on the summary provided but now including an assessment of the potential change in price for tomorrow.
            
            If you find any trends or indicators suggesting a certain direction for tomorrow's price change, make sure to provide supporting evidence.
            
            Make sure to use the most recent data available.
            {self.__tip_section()}
        """),
        agent=agent
    )

  def filings_analysis(self, agent):
    logging.info(f"{agent} is performing filings analysis task")
    return Task(
      description=dedent(f"""
          Analyze the latest 10-Q and 10-K filings from EDGAR for
          the {self.company}. 
          Focus on key sections like Management's Discussion and
          Analysis, financial statements, insider trading activity, 
          and any disclosed risks.
          Extract relevant data and insights that could influence
          the stocks' future performance, including recent insider trades and other changes.
  
          Your final answer must be an expanded report that now
          also highlights significant findings from these filings,
          including any red flags or positive indicators for
          your customer, with a specific focus on potential impacts on tomorrow's price.
          {self.__tip_section()}
      """),
      agent=agent
    )


  def classify_stock(self, agent):
      logging.info(f"{agent} is performing stock classification task")
      return Task(
          description=dedent(f"""
              Classify the stock based on the collected data, focusing on the potential change in price for tomorrow.
              Utilize the insights gathered from research, financial analysis, filings analysis, and other relevant data also collect the links and date of publish.
              
              Define the following classes:
              - Class 0: Indicates that the stock is expected to decrease in price tomorrow.
              - Class 1: Indicates that the stock is expected to remain relatively stable in price tomorrow.
              - Class 2: Indicates that the stock is expected to increase in price tomorrow.
              
              Define thresholds for classification:
              - If the predicted percentage change in price is less than -1, classify it as Class 0.
              - If it falls between -1 and 1, classify it as Class 1.
              - If it is greater than 1, classify it as Class 2.
              
              Your final answer must be in JSON format, containing:
              - Ticker: {self.company}
              - Classification: Class 0, Class 1, Class 2
              - Confidence Score: [0.0, 1.0]
              - Reasoning for the classification
              - supporting links and articles
              
              Make sure to provide a clear explanation for the classification, along with the confidence score.
              Ensure the recommendation is based on thorough analysis and supporting evidence.
              {self.__tip_section()}
          """),
          agent=agent
    )

  def __tip_section(self):
    return "If you do your BEST WORK, I'll give you a $10,000 commission!"
