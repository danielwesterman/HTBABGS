#!/usr/bin/env python
# coding: utf-8

# In[38]:


import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression


# In[39]:


import pandas as pd
cheat = pd.read_csv('cheat.csv')
cheat2 = pd.read_csv('cheat2.csv')
cheat2 = cheat2.set_index('Episode')
len(cheat[cheat['Cheated'] == True])/len(cheat[cheat['Correct'] == True])
cheat


# In[40]:


cheat['Cheated'].sum()/cheat['Correct'].sum()


# In[41]:


def Player_Round(Episode, Round, Player, df):
    df = df[df.Episode == Episode]
    df = df[df.Round == Round]
    
    opponentTotalCorrect= 0
    opponentTotalDecisions = 0
    for player in df['Player'].unique():
        if player != Player:
            df2 = df[df.Player != player]
            df2_cheated = df2[df2.Cheated == True]
            df2_honest = df2[df2.Cheated == False]
            for accuser in df2_honest.Accused:
                if player in accuser:
                    opponentTotalDecisions = opponentTotalDecisions + 1
            for accuser in df2_cheated.Accused:
                if player in accuser:
                    opponentTotalDecisions = opponentTotalDecisions + 1
                    opponentTotalCorrect = opponentTotalCorrect + 1
            for accuser in df2_honest.Accused:
                if player not in accuser:
                    opponentTotalDecisions = opponentTotalDecisions + 1
                    opponentTotalCorrect = opponentTotalCorrect + 1
            for accuser in df2_cheated.Accused:
                if player not in accuser:
                    opponentTotalDecisions = opponentTotalDecisions + 1
                
    opponentTotalAccuracy = opponentTotalCorrect/opponentTotalDecisions
    
    
    
    df1 = df[df.Player == Player]
    df = df[df.Player != Player]
    df_cheated = df[df.Cheated == True]
    df_honest = df[df.Cheated == False]
    
    
    opponent_cheats = len(df_cheated)
    
    wrong = 0 
    for i in df_honest.Accused:
        if Player in i:
            wrong = wrong + 1
        
    right = 0
    for i in df_cheated.Accused:
        if Player in i:
            right = right + 1
            
    deferred_correct = 0
    for i in df_honest.Accused:
        if Player not in i:
            deferred_correct = deferred_correct + 1
        
    deferred_incorrect = 0
    for i in df_cheated.Accused:
        if Player not in i:
            deferred_incorrect = deferred_incorrect + 1
            
    accuses = right+wrong
    
    if right+wrong>0:
        accuse_accuracy = (right/(right+wrong))
    else:
        accuse_accuracy = 0
    
    
    total_accuracy = (right + deferred_correct)/(right + wrong + 
                                                    deferred_correct + deferred_incorrect)
    accuse_plus_minus = right - wrong
    total_correct = right + deferred_correct
    choices = right + wrong + deferred_correct + deferred_incorrect
    
    winner = Player == cheat2.loc[Episode]["Detector " + str(Round)]
    
    accuse_rate = accuses/choices
    
    counts = df1['Cheated'].value_counts()
    
    cheats = counts.get(True, 0)
    
    correct = df1['Correct'].values.sum()
    
    honest = counts.get(False, 0)
    
    cheat_rate = cheats/correct    
    
    
    
    return {'Episode': Episode,
            'Round': Round,
        'Player' : Player,
        'Total Correct' : total_correct,
        'Choices': choices,
            'Accuses': accuses,
        'True Positive': right,
       'False Positive': wrong,
       'True Negative': deferred_correct,
       'False Negative': deferred_incorrect,
           'Accuse +-': accuse_plus_minus,
           'Accuse Accuracy': round(accuse_accuracy,2),
           'Total Accuracy': round(total_accuracy,2),
            'Accuse Rate' : round(accuse_rate, 2),
            'Correct' : correct,
            'Cheats' : cheats,
            'Non-Cheats' : honest,
            'Cheat Rate' : round(cheat_rate, 2),
            'Opponent Cheats' : opponent_cheats,
            'Opponent Total Accuracy' : round(opponentTotalAccuracy, 2),
           'Winner' : winner}


# In[43]:


def Round(Episode, Round):
    one = cheat2.loc[Episode]["First Place"]
    two = cheat2.loc[Episode]["Second Place"]
    three = cheat2.loc[Episode]["Third Place"]
    four = cheat2.loc[Episode]["Fourth Place"]
    if Round == 2:
        return Player_Round(Episode, Round, one, cheat), Player_Round(Episode, Round, two, cheat), Player_Round(Episode, Round, three, cheat)
    return Player_Round(Episode, Round, one, cheat), Player_Round(Episode, Round, two, cheat), Player_Round(Episode, Round, three, cheat), Player_Round(Episode, Round, four, cheat)


# In[44]:


Round(1,2)


# In[45]:


combinedTuple = ()

for episodes in range(1,13):
    for rounds in range(1,3):
        combinedTuple = combinedTuple + Round(episodes, rounds)
        
player = pd.DataFrame(combinedTuple)
player['TotalRound'] = ((player['Episode'] - 1) * 2) + player['Round']
round_winner = player[player['Winner'] == True]
round_loser = player[player['Winner'] == False]
print(player['Accuses'].sum()/player['Choices'].sum())
average_total_accuracy = (player['Total Correct'].sum()/player['Choices'].sum())
round_winner['Total Accuracy'].mean()


# In[46]:


player[player['Cheats'] > 2]


# In[47]:


#player[player['Accuses'] == 0]
# round_winner[round_winner['TotalRound'] == 16]
# player[player["Episode"] == 9]


# In[48]:


labeled_ticks = [(x*2-1) for x in range(1,13)]
empty_ticks = [(x*2) for x in range(1,13)]

ax = round_winner.plot(x = 'TotalRound', y = 'Total Accuracy', color = 'green', xticks= labeled_ticks, title = "Total Accuracy: Winner vs Losers")


round_loser.plot.scatter(x = 'TotalRound', y = 'Total Accuracy', ax = ax, color = 'red')

ax.set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
ax.set_xticks(empty_ticks, minor = True)
ax.set_xlabel('Episode')
ax.legend(labels = ['Winner', 'Losers'])
plt.savefig('TotalAccuracy.png')


# In[49]:


labeled_ticks = [(x*2-1) for x in range(1,13)]
empty_ticks = [(x*2) for x in range(1,13)]

round_winner_1 = round_winner[round_winner['TotalRound'] < 15]
round_winner_2 = round_winner[round_winner['TotalRound'] >15]
ax2 = round_winner_1.plot(x = 'TotalRound', y = 'Accuse Accuracy', color = 'green', xticks= labeled_ticks,title = "Accuse Accuracy: Winner vs Losers", label = "Winner")
round_winner_2.plot(x = 'TotalRound', y = 'Accuse Accuracy', color = 'green', ax = ax2, label = "Winner")

round_loser.plot.scatter(x = 'TotalRound', y = 'Accuse Accuracy', ax = ax2, color = 'red', label = "Losers")

ax2.set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
ax2.set_xticks(empty_ticks, minor = True)
ax2.set_xlabel('Episode')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles[1:3], labels=labels[1:3], loc='upper right', bbox_to_anchor=(0.9, 1))
plt.savefig('AccuseAccuracy.png')


# In[50]:


labeled_ticks = [(x*2-1) for x in range(1,13)]
empty_ticks = [(x*2) for x in range(1,13)]

ax3 = round_winner.plot(x = 'TotalRound', y = 'Accuse +-', color = 'green', xticks= labeled_ticks, title = "Accuse +-: Winner vs Losers")


round_loser.plot.scatter(x = 'TotalRound', y = 'Accuse +-', ax = ax3, color = 'red')
ax3.set_xticks(empty_ticks, minor = True)
ax3.set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
ax3.set_xlabel('Episode')
ax2
ax3.legend(labels = ['Winner', 'Losers'], loc = 'upper right', bbox_to_anchor = (.95,1))
plt.savefig('Accuseplusminus.png')


# In[51]:


ax4 = round_winner.plot.scatter(x = 'Accuse Rate', y = 'Total Accuracy', c = 'green', title = 'Accuse Rate vs Total Accuracy')
round_loser.plot.scatter(x = 'Accuse Rate', y = 'Total Accuracy', c = 'red', ax = ax4)
ax4.legend(labels = ['Winner', 'Loser'])


# In[52]:


ax5 = round_loser.plot.scatter(x = 'Cheats', y = 'Opponent Total Accuracy', c = 'red', title = 'Numeber of Cheats vs Opponent Total Accuracy')
round_winner.plot.scatter(x = 'Cheats', y = 'Opponent Total Accuracy', c = 'green', ax = ax5)
ax5.legend(labels = ['Winner', 'Loser'])
ax5.axhline(y=average_total_accuracy, color='blue', linestyle='--')


# In[53]:


ax6 = round_winner.plot.scatter(x = 'Opponent Total Accuracy', y = 'Total Accuracy', c = 'green', title = 'Total Accuracy and Opponet Total Accuracy: Winners and Losers')
round_loser.plot.scatter(x = 'Opponent Total Accuracy', y = 'Total Accuracy', c = 'red', ax = ax6)
ax6.legend(labels = ['Winner', 'Loser'])


# In[54]:


labeled_ticks = [(x*2-1) for x in range(1,13)]
empty_ticks = [(x*2) for x in range(1,13)]

ax7 = round_winner.plot(x = 'TotalRound', y = 'False Positive', color = 'green', xticks= labeled_ticks, title = "False accusations: Winner vs Losers")


round_loser.plot.scatter(x = 'TotalRound', y = 'False Positive', ax = ax7, color = 'red')
ax7.set_xticks(empty_ticks, minor = True)
ax7.set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
ax7.set_xlabel('Episode')
ax7.set_ylabel('False Accusations')
ax7.legend(labels = ['Winner', 'Losers'], loc = 'upper right', bbox_to_anchor = (1,1))


# In[55]:


labeled_ticks = [(x*2-1) for x in range(1,13)]
empty_ticks = [(x*2) for x in range(1,13)]

ax8 = round_winner.plot(x = 'TotalRound', y = 'True Positive', color = 'green', xticks= labeled_ticks, title = "True accusations: Winner vs Losers")


round_loser.plot.scatter(x = 'TotalRound', y = 'True Positive', ax = ax8, color = 'red')
ax8.set_xticks(empty_ticks, minor = True)
ax8.set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
ax8.set_xlabel('Episode')
ax8.set_ylabel('True Accusations')
ax8.legend(labels = ['Winner', 'Losers'], loc = 'upper right', bbox_to_anchor = (1,1))


# In[56]:


ax9 = round_winner.plot.scatter(x = 'Cheats', y = 'Total Accuracy', c = 'green', title = 'Cheats vs Own Total Accuracy')
round_loser.plot.scatter(x = 'Cheats', y = 'Total Accuracy', c = 'red', ax = ax9)
ax9.legend(labels = ['Winner', 'Loser'])
ax9.set_xticks([0,1,2,3,4])


# In[57]:


X = player[['Cheats']]
y = player['Opponent Total Accuracy']

lin = LinearRegression()
lin.fit(X, y)
lin.coef_


# In[ ]:





# In[ ]:




