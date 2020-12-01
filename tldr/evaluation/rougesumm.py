from rouge_score import rouge_scorer

text = '''
(CNN)Legal experts have been saying for a week now that President Donald Trump's court cases to throw out ballots and turn around his election loss were bound to fail.

Throughout Friday, the failures piled up.
In one day, nine cases meant to attack President-elect Joe Biden's win in key states were denied or dropped, adding up to a brutal series of losses for the President, who's already lost and refuses to let go. Many of the cases are built upon a foundational idea that absentee voting and slight mismanagement of elections invite widespread fraud, which is not proven and state leaders have overwhelming said did not happen in 2020.
In court on Friday:
The Trump campaign lost six cases in Montgomery County and Philadelphia County in Pennsylvania over whether almost 9,000 absentee ballots could be thrown out.
The Trump campaign dropped a lawsuit in Arizona seeking a review by hand of all ballots because Biden's win wouldn't change.
A Republican candidate and voters in Pennsylvania lost a case over absentee ballots that arrived after Election Day, because they didn't have the ability to sue. A case addressing similar issue is still waiting on decisions from the Supreme Court -- which has remained noticeably silent on election disputes since before Election Day.
Pollwatchers in Michigan lost their case to stop the certification of votes in Detroit, and a judge rejected their allegations of fraud.
On top of it all, a law firm leading the most broad challenge in Pennsylvania -- perhaps the most significant state for Trump's post-election fight -- dropped out.
"The Trump campaign keeps hoping it will find a judge that treats lawsuits like tweets," said Justin Levitt, a Loyola Law School professor and elections law expert, on Friday. "Repeatedly, every person with a robe they've encountered has said, 'I'm sorry, we do law here.'"
And yet, lawyers representing, Republicans and voters unhappy with the election's result forge ahead, as part of an increasingly desperate long-shot attempt to swing the Electoral College in Trump's favor, no matter the popular vote and electoral count victory for Biden.
'''

with open ("summ.txt", "r") as myfile:
    summ=myfile.readlines()

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) 
scores = scorer.score(text, summ[0])

print("---\n")
print(scores)
