import pickle
from analiseSentimentos import clean, is_special, to_lower, rem_stopwords,stem_txt
# Carrega o vocabulário e o modelo treinado
cv = pickle.load(open('cv.pk1', 'rb'))
bnb = pickle.load(open('model0.pk1', 'rb'))

rev = """
More so than with any other film I've watched in recent memory, I rather don't entirely know what to say after watching this. There's a stark, 
jolting precision and brutality to much of the movie, even well preceding the abject violence, yet also a weirdly poetic beauty at many points. 
This is as true for Lindsay Anderson's direction as it is for David Sherwin's screenplay, in all ways, but also for the acting. Performances are 
exact and practiced, but also fluid and natural. Whether presented in pure black and white, or in color under drab skies or sunny blue, the very 
image before us and cinematography is rich and lush. For viewers such as myself whose perspective on schooling in the United Kingdom is informed 
wholly by cinematic exhibition and not personal experience, the strict regimentation and forced social arrangements are both fascinating and 
uninviting - to say nothing of what embellishments the movie makes in imparting its tale. When all is said and done, the result is that for any 
similarities one could find to this, that, or the other thing in more than 50 years since, 'If....' still feels quite unlike anything else.

For all the pomp and circumstance and plays for power and social position, and the inherent fictional nature of the feature, there's an earnestness 
to every aspect - characters, dialogue, scene writing, narrative, direction, performances - that comes off as very real, organic and relatable. It's 
an enticing balance maintained at all times between various moods and tones, with the interactions between characters taking foremost precedence as 
a focal point and anchor, whether trending toward antagonism or camaraderie. And with that said, not to belabor the point, but the contributions of 
the actors seems particularly essential in 'If....' to cementing the picture. As I've suggested, I think everyone on hand does a fine job of helping 
to bring the story to life with portrayals of nuance, poise, and personality, yet this goes above all for those whose characters are ultimately dubbed 
the "crusaders." While credited alongside those more prominent, Rupert Webster and Christine Noonan have little more than bit parts as Philips and 
"the girl"; we know so little about their characters, and one wishes they could have been fleshed out more at least to solidify motivations. Still, 
Webster and Noonan make strong impressions despite their limited time on screen. David Wood and Richard Warwick are decidedly more visible as Knightly 
and Wallace, and both actors do well in embodying the sneering disregard of the boys. But of course it's unmistakable Malcolm McDowell, starring as 
protagonist Mick Travis, who stands out most of all. There are subtleties in McDowell's distinct vocal timbre, and in his expressions and body language, 
that communicate definite confidence, defiance, and attitude, and just as it's hard to imagine anyone else as Alex DeLarge in 'A Clockwork Orange,' he 
is a perfect fit to depict the boiling malcontent of young Travis.

I don't feel that it's perfect. As well made as it is, and as enjoyable as the viewing experience is, there's a part of me that think maybe my perception 
of shortcomings is actually just an inability to glean the artistic choice behind certain inclusions. Again speaking to the characters of Philips and the 
girl - we're given minimal information of them generally, and little or nothing that would meaningfully serve to explain their participation in the finale. 
Jute is given a fair amount of screen time early on, then wordlessly fades from the narrative. One could infer to a reasonable certainty the significance 
of a specific scene featuring Mrs. Kemp, but in the end it just seems superfluous to the whole. Broadly speaking, it just seems like the writing could have 
stood to be a little tighter and more concrete; by no means does this completely dampen the value, but it's a notable aspect of the production.

Subjective faults notwithstanding, however - by and large, 'If....' is pretty fantastic. I'm not sure that it totally met my expectations based on what 
little I had read of it, but for the most part, I'm glad to have been surprised. It's a wonderfully subversive story of individuality and discontent set 
against the rigidity and corruption of the establishment, and it's presented with a refined touch behind almost every element. Even if something about the 
feature feels a little off, and not fully copacetic, that sense is minor in comparison to the engrossing drama to play out. Minding content warnings for 
violence and nudity, this isn't going to be for everyone, but I think it's solid enough that I'd have no qualms about recommending it to just about anyone. 
Though perhaps not altogether essential, 'If....' is an excellent, satisfying picture that's worth checking out if one has the opportunity.
"""
f1 = clean(rev)

f2 = is_special(f1)
f3 = to_lower(f2)
print('F1: ', f3)
f4 = rem_stopwords(f3)
f5 = stem_txt(f4)


inp =  cv.transform([f5]).toarray()
y_pred = bnb.predict(inp)

print("predict teste: ", y_pred)