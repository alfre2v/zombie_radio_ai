# zradio_local

Execution Notes

Simple Python script that achieves what I want all locally in my aorusX570. This assumes local accessibility to ollama, F5-TTS, and possibly Whisper.

What I want for this local prototype stage:

1. It is a light local python process that relies on asyncio/threads to execute tasks concurrently.
2. It can play audio to the PC speakers.
3. It can record audio from the PC microphone.
4. It uses yaml for its configuration file.
5. It has simple instructions to interact with several LLMs and prompt them to play character parts in a "theater play" about Zombies and Halloween.
6. Each character in the play has its own distinct voice with clear emotional states.
7. Optionally, the process also listens to what the users say to the microphone and interact and the LLM characters can respond to the user voice.












## How to prompt the LLM to write the story


### Attempt with model llama3.2: ollama run llama3.2:latest

To be conducted.


### Attempt with model nemotron-mini: ollama run nemotron-mini


```text
>>> Write a story about zombies and halloween about 4 characters in an underground research laboratory where an accident happened to release
...  a deadly virus that turn people into zombies. The 4 characters are trapped down in the lab and cannot escape due to security protocols. 
... I describe some elements of each character of the story: 1. Dr. Edwin Kund, a man in his early 50s, head of the lab, he has all the secre
... ts and is the one more directly responsible for the disaster; 2. Dr. Hanna Brie, a woman in her late 30s, research biologist specialized
...  in exotic virus; 3. Miss Betty Campbell, a young woman in her early 20s, she is the secretary of Dr. Kund; and 4. Dr. Alfred Valles, nuc
... lear physicist responsible for the nuclear reactor powering the lab and for the multiple gamma ray sources used in the lab to induce acce
... larated mutations in viruses. Make the story narrated from the perspective of Dr. Alfred Valles, i.e. narrated in first person. Try to ma
... ke the story scary but also fun.
 Sure, I'd be happy to help you with that! Here's a short horror-themed story based on your description:

---

I, Dr. Alfred Valles, am trapped down here in this eerie underground laboratory. The air is thick and stale, filled with the smell of 
dust and decay, but I can't blame it all on these conditions - they are what kept me alive during my time working on this project. 
However, the silence that surrounds us now feels like a tomb, an endless mausoleum where any hope for escape has been silenced by those 
very conditions. 

We had just finished our experiments with the new gamma-ray accelerator last night when things took a horrific turn for the worst. The 
machine was not functioning properly and we couldn't get it back online in time before Dr. Edwin Kund closed down the facility, fearing 
that something terrible might happen to us if he left it running unattended for too long. 

The next day dawned with no sign of life outside - all communication lines were down due to some kind of electronic warfare or 
accident. I couldn't even tell my family back home what happened here as we were trapped and isolated by the virus that had been 
released during our experiments, a deadly one that would soon transform everyone into zombies once it reached them. 

We managed to secure ourselves in this underground lab because Dr. Kund himself set up security protocols before he left for his last 
assignment - an unfortunate move if you ask me now as we were trapped down here with no way out. The only person I can rely on is Miss 
Betty Campbell, our secretary and the most level-headed member of our team despite her young age. 

Dr. Hanna Brie, my colleague from the research department, had gone mad long ago when she realized what was happening - or should have 
happened if things had gone as planned. She kept trying to find a way out but without success. Now, I can't even imagine how she held 
on this long given her state of mind and our situation here. 

Dr. Edwin Kund himself is responsible for everything that went wrong - he was the one who decided to proceed with these experiments 
despite knowing the risks involved. He claims it all had a purpose and we would have made great strides in science if things had worked 
out as planned, but I can't say I agree or believe him anymore after what has happened here. 

I can only hope that Miss Campbell will find an escape route for us soon - maybe she knows something about this lab that no one else 
does and could potentially help us get out of here before we become the next meal for these creatures. I'll keep trying to figure out 
if there was anything more we could have done differently during our experiments, but it feels like a futile effort now as everything 
is spiraling down around us into darkness and decay. 

I hope that this story doesn't become true - my family needs me alive for some time yet and I can only pray to whatever god or gods 
there are out there in the world above ground that we find our way out of here soon. But until then, I must accept this fate as my own 
curse because it was my responsibility as a scientist and a researcher to bring these experiments to fruition - even if they came at 
such a terrible price for us all. 

---

I hope you enjoy the story! Let me know if there's anything else I can help you with.

>>> There is an update in the story, Dr. Hanna Brie died, horribly bitten by the zombies when she tried to escape. Write a continuation of th
... e story that incorporate this fact.
 Sure, here's how the story could continue: 

---

I found out about it later on from Miss Campbell - she had managed to get in contact with someone outside and was able to pass along 
some information about our situation. Dr. Hanna Brie, despite her worsening state of mind due to what happened in that lab earlier, had 
tried to escape through a ventilation shaft when we thought things were just taking longer than expected as the result of an equipment 
malfunction. But she never made it out alive - she was found horribly bitten and torn apart by zombies once they reached us down here.

Miss Campbell is devastated by this news but also determined to find a way out for herself and me after Dr. Hanna Brie's death. She 
managed to secure some information from the person she had contacted outside about our situation here, including details of our 
security protocols and the location of the ventilation shafts we hadn't found yet due to our isolation down here in this part of the 
lab with no one else around for quite some time now that Dr. Hanna Brie was gone. 

Dr. Edwin Kund, while he seems less concerned about his own safety at times (he has been more focused on trying to contain and 
understand what had happened rather than worrying about us or our situation), is still trying to figure out if there's any way we can 
escape from down here alive with the information Miss Campbell managed to secure. But it feels like a losing battle now that she was 
also gone - he couldn't have foreseen this happening, but I guess these things happen when you try and push boundaries too much in your 
work as well. 

I just hope we can find an escape route soon for both of us before the zombies start swarming out here looking for more food or 
something like that. Miss Campbell is trying to figure it all out with whatever knowledge she has gained from outside, while I'm still 
trying to come up with any sort of solution based on our experiments and what we know about each other - but time feels like its 
running short as nothing seems to be working so far even after everything had been done. 

I can only hope that there are others out there who will try to help us or maybe someone from the outside will find this story of ours 
and figure something out before it's too late for both of us - I really don't want anyone else to go through what we did down here in 
this lab. But until then, all I can do is try and make sure that Miss Campbell knows everything she needs to know about our situation 
here so that she could potentially help herself or me out if she manages to find someone outside who might be able to help us as well. 

---

I hope you enjoy the continuation! Let me know if there's anything else I can assist you with.

```
