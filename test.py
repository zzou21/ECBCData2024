import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
text = ""
punkt_param = PunktParameters()
sentence_tokenizer = PunktSentenceTokenizer(punkt_param)

sentences = sentence_tokenizer.tokenize(text)

for i, sentence in enumerate(sentences):
    print(f"Sentence {i+1}: {sentence}")


# from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters


# # # Load the MacBERTh tokenizer and model
# # tokenizer = AutoTokenizer.from_pretrained("emanjavacas/MacBERTh")
# # model = AutoModelForTokenClassification.from_pretrained("emanjavacas/MacBERTh")

# # Example paragraph in Early Modern English
# text = "But aye, thereby shall safe and quiet rest, As swallowes which besides is build their nest. Oh may this fire within these walls indure, So long as Neptunes waues this Ile immure: And as from Mountaines comes that wholesome breath, Which healthfull makes the Valleys all beneath: So from this Bi∣shop Mon∣taigne MONTAIGNE health come and saluation, Vnto the Founder, and his generation: Let Prophets, Priests, in Prayers all combine To make this House a Blisse to Thee and Thine: And when by their deuotions ioy'nd, this flame Is kindled; let thy Priests maintaine the same And offer vp thy prayers day and night, Like fumes of Incense, in th' Almighties sight; Oh force vnited, of a Congregation That ioyne in prayer at a Consecration; With these my Muse (now thine) shall beare a part, And whilst they pray by booke, shee'le pray by hart. FINIS. THE CONVERTS CONQVEST. PSAL. 119. V. Ultimo. I haue gone astray like a sheepe that is lost: oh seeke thy seruant, for I doe not forget thy Commandements. LEast I be deem'd a thiefe, I will disclose; I turn'd to Verse what you gaue me in Prose: In so few lines I neuer yet did find More heav'nly Comforts to a grieued mind: Mans sinfull Heart, Hells Malice, Grace diuine, Is intermixed so in ev'ry Line. I praise God, I this speake with feeling Sense, God grant the Reader like experience! Good publish't, doth more good, by being knowne, Wherein I seeke Gods glory, not mine owne: Of Reading and of Writing's but one end, Repent, beleeue, for sake Sinne and amend. Your true vnfeigned Friend, R. A. THE CONVERTS CONQVEST. ACertaine Christian which had often bin Tempted, and by his weaknesse ouertaken To his great sorrow, with one selfe same sin, At last sate downe as if he were forsaken; Where of sins bondage first he doth complaine, And then himselfe thus comforteth againe: From my all-seeing God I cannot flye, Still in my loathed sins I may not lye; Yet ought I not of mercy to despaire, Yet dare I not for Grace to God repaire▪ Pray would I, but I cannot it intend; Repent I doe, but not my life amend: I to beleeue desirous am, yet doubt, In this lewd wicked custome holding out; God is by me dishonoured, whilst I Professe to serue him true and faithfully: Disples'd, whilst I prouoke him to his face; Both griev'd and quenched is his Spirit of Grace; His Graces I abated, withered, find, My sense benum'd; besotted is my mind, My memory dull'd, more strong grows Satans dart, My Soule's aduentred, hardned is my hart; I grow in Sin rich, poore in Goodnesse, Grace, My head's vex'd, Conscience is in wofull case; My calling stain'd, crack'd credit, Time mispended, My strength consum'd, and my God offended: As doth my sin, my burthen doth increase; My pain's inlarged, troubled is my peace. I sigh, but sorrow not aright, would faine Be rid of it, but soone returne againe: I grieue, not weepe; Lord! could I from it part, Forsake, confesse it with a broken hart. How farre aduenture, Lord! how long shall I Dare to prouoke thy powerfull Maiesty? How long shall he forbeare? how often might He cut mee off? Or suddenly dead smite, How long shall hee chastise mee, yet in vaine? At length, O Lord, be mercifull againe: Oh tarry not, Lord, tarry not too long, But make my resolution firme and strong. Oh loathsomenesse, deceitfulnesse of sin, Sweetnesse, and bitternesse wee find therein; Beginnings, fawnings, growing, terrour, smart, Faiths weakenesse; Satans enuy, Mans false heart: When shall I now these? Oh that I could know Them better, Lord! by farre than yet I doe: Yet wish (though much asham'd thus to be tainted) I were not as I am with them acquainted. What shall I doe? Goe on! Nay, God defend! Shall I retire? Stand idle? Not amend? Shall I despaire? Why so? Haue my sins quite Dride vp Gods mercies which are infinite? Such thing to thinke, were cursed Blasphemy, Who succours all that are in misery: Will not God heare what I in Faith desire? Humbled with Griefe? Then make I him a Lier. Shall I presume yet longer? Ah I haue Presum'd too much: Oh let mee mercy craue, By true Repentance, and abundant teares; What is thy heart so harden'd, as it feares It neuer can be mollify'd againe? Then Gods Omnipotence thou dost restraine: What? Hath this thing without God come to passe? Hath Satan got the Victory? Alas! Is not th'Almightie far nore strong than hee? Hath not my Lord, Christ Iesus dy'd for mee? Hath God ere lov'd thee? Sure hee once me lov'd, For then I it by good experience prov'd; Then Loues he still, for where he doth begin Hee loues for euer, and his gifts haue bin Without Repentance: hee for mee destry'd And vanquisht Death, Sin, Satan when he dy'd. O Lord encrease my faith, why should not I Beleeue in him, obey him willingly? How faine would I beleeue, and him obay; How fame would I repent, amend, and pray: I cannot then conclude, nor will, nor dare, That I am damn'd, for these desires sure are The motions of Gods Spirit in mee indeed; Who neither smoaking flax, nor bruised reed Will quench, or breake, But all will satisfie Who thirst and hunger after equitie. Blist euer be his name who hath begun, To make me Conqueror through Christ his Sonne. By his assistance gracious, then I Vow To serue God better then I haue, till now, On his behests more carefully attend, Thy Grace mee strengthen, as a sheild defend. Satan auoid, thou hast in mee no part, From the beginning thou a Liar art; Before and after mine, in Adams fall, Thou to deceiue mee practisest in all: But God is true, iust, mercifull to mee In Iesus Christ his blessed Sonne and hee, For honour of his Name and Maiesty, Will doe away all mine iniquitie: So as the siftings here of Satan shall Not turne to my destruction; But they all Gods Grace in mee shall further magnifie And bind mee to him more assuredly; More hee forgiues, the greater is his grace, Him faster we with Loue in Christ imbrace. Henceforth my soule remember well, what gaine Thou reaped hast, and oft maist reape againe, By that whereof thou iustly art ashamed, For which thy Name and Conscience now is blamed. Restore me to the Ioy of thy Saluation, Which better is then ioyes continuation; For by the want, the worth discerne we may, And be stirr'd vp more earnestly to pray."

# # Tokenize the input text
# tokens = tokenizer(
#     text,
#     return_tensors="pt",
#     truncation=True,
#     padding="max_length",  # Ensures all sequences are padded/truncated to the same length
#     max_length=30,  # Example length, adjust based on your needs and model constraints
#     is_split_into_words=False
# )

# # Get predictions from the model
# with torch.no_grad():
#     outputs = model(**tokens)

# # Get the token classification predictions
# predictions = torch.argmax(outputs.logits, dim=2)

# # Decode tokens
# decoded_tokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])

# # Print tokens with their corresponding label indices
# for token, prediction in zip(decoded_tokens, predictions[0]):
#     print(f"Token: {token}, Label index: {prediction.item()}")

# # Assuming label index 1 indicates the end of a sentence based on the given example
# sentence_boundary_label = 1

# # Reconstruct sentences based on token classification predictions
# sentences = []
# current_sentence = []

# for token, prediction in zip(decoded_tokens, predictions[0]):
#     word = tokenizer.decode([tokenizer.convert_tokens_to_ids(token)])
#     if token in ("[CLS]", "[SEP]"):
#         continue  # Skip special tokens
#     current_sentence.append(word)
#     if prediction.item() == sentence_boundary_label:
#         sentences.append(' '.join(current_sentence).replace(' ##', ''))
#         current_sentence = []  # Start a new sentence

# # Add the last sentence if there are any remaining tokens
# if current_sentence:
#     sentences.append(' '.join(current_sentence).replace(' ##', ''))

# # Print the extracted sentences
# for i, sentence in enumerate(sentences):
#     print(f"Sentence {i+1}: {sentence}")