---
license: mit
language:
- zh
- en
pipeline_tag: text-generation
library_name: transformers
---

# GLM-4-9B-0414

## Introduction

The GLM family welcomes new members, the **GLM-4-32B-0414** series models, featuring 32 billion parameters. Its performance is comparable to OpenAI’s GPT series and DeepSeek’s V3/R1 series. It also supports very user-friendly local deployment features. GLM-4-32B-Base-0414 was pre-trained on 15T of high-quality data, including substantial reasoning-type synthetic data. This lays the foundation for subsequent reinforcement learning extensions. In the post-training stage, we employed human preference alignment for dialogue scenarios. Additionally, using techniques like rejection sampling and reinforcement learning, we enhanced the model’s performance in instruction following, engineering code, and function calling, thus strengthening the atomic capabilities required for agent tasks. GLM-4-32B-0414 achieves good results in engineering code, Artifact generation, function calling, search-based Q&A, and report generation. In particular, on several benchmarks, such as code generation or specific Q&A tasks, GLM-4-32B-Base-0414 achieves comparable performance with those larger models like GPT-4o and DeepSeek-V3-0324 (671B).

**GLM-Z1-32B-0414** is a reasoning model with deep thinking capabilities. This was developed based on GLM-4-32B-0414 through cold start, extended reinforcement learning, and further training on tasks including mathematics, code, and logic. Compared to the base model, GLM-Z1-32B-0414 significantly improves mathematical abilities and the capability to solve complex tasks. During training, we also introduced general reinforcement learning based on pairwise ranking feedback, which enhances the model's general capabilities.

**GLM-Z1-Rumination-32B-0414** is a deep reasoning model with rumination capabilities (against OpenAI's Deep Research). Unlike typical deep thinking models, the rumination model is capable of deeper and longer thinking to solve more open-ended and complex problems (e.g., writing a comparative analysis of AI development in two cities and their future development plans). Z1-Rumination is trained through scaling end-to-end reinforcement learning with responses graded by the ground truth answers or rubrics and can make use of search tools during its deep thinking process to handle complex tasks. The model shows significant improvements in research-style writing and complex  tasks.

Finally, **GLM-Z1-9B-0414** is a surprise. We employed all the aforementioned techniques to train a small model (9B). GLM-Z1-9B-0414  exhibits excellent capabilities in mathematical reasoning and general tasks. Its overall performance is top-ranked among all open-source models of the same size. Especially in resource-constrained scenarios, this model achieves an excellent balance between efficiency and effectiveness, providing a powerful option for users seeking lightweight deployment.

## Showcase

### Animation Generation

<table>
  <tr>
    <td style="text-align: center; font-size: 16px; font-weight: bold; padding: 10px; width: 420px;">
      GLM-Z1-32B-0414
    </td>
    <td style="text-align: center; font-size: 16px; font-weight: bold; padding: 10px; width: 420px;">
      GLM-4-32B-0414
    </td>
  </tr>
  <tr>
    <td style="vertical-align: top; padding: 10px; width: 420px;">
      <video src="https://github.com/user-attachments/assets/849ff9fd-b54d-4c74-9ee5-3412e1a09e32"
             style="width: 400px; height: 300px; object-fit: contain;" autoplay loop muted playsinline></video>
      <div style="margin-top: 10px; font-size: 14px; color: #333; width: 400px;">
        write a Python program that shows a ball bouncing inside a spinning hexagon. The ball should be affected by gravity and friction, and it must bounce off the rotating walls realistically
      </div>
    </td>
    <td style="vertical-align: top; padding: 10px; width: 420px;">
      <video src="https://github.com/user-attachments/assets/8dccdb9d-cc44-4732-b438-74a4e3cb9dfb"
             style="width: 400px; height: 300px; object-fit: contain;" autoplay loop muted playsinline></video>
      <div style="margin-top: 10px; font-size: 14px; color: #333; width: 400px;">
         Use HTML to simulate the scenario of a small ball released from the center of a rotating hexagon. Consider the collision between the ball and the hexagon's edges, the gravity acting on the ball, and assume all collisions are perfectly elastic. (Prompt translated from Chinese)
      </div>
    </td>
  </tr>
</table>

### Web Design

<table>
  <tr>
    <td style="text-align: center; font-size: 16px; font-weight: bold; padding: 10px; width: 420px;">
      GLM-4-32B-0414
    </td>
    <td style="text-align: center; font-size: 16px; font-weight: bold; padding: 10px; width: 420px;">
      GLM-4-32B-0414
    </td>
  </tr>
  <tr>
    <td style="vertical-align: top; padding: 10px; width: 420px;">
      <img src="https://github.com/user-attachments/assets/bd9c1fc1-c784-4e8f-9c76-5f7389a715f1"/>
      <div style="margin-top: 10px; font-size: 14px; color: #333; width: 400px;">
          Design a drawing board that supports custom function plotting, allowing adding and deleting custom functions, and assigning colors to functions. (Prompt translated from Chinese)
      </div>
    </td>
    <td style="vertical-align: top; padding: 10px; width: 420px;">
      <img src="https://github.com/user-attachments/assets/7ad12d52-9229-4278-8d1b-ffbf43e99070"/>
      <div style="margin-top: 10px; font-size: 14px; color: #333; width: 400px;"> Design a UI for a mobile machine learning platform, which should include interfaces for training tasks, storage management, and personal statistics. The personal statistics interface should use charts to display the user's resource usage over a period. Use Tailwind CSS to style the page, and display these 3 mobile interfaces tiled on a single HTML page. (Prompt translated from Chinese) </div>
    </td>
  </tr>
</table>

### SVG Generation

<table>
  <tr>
    <td style="text-align: center; font-size: 16px; font-weight: bold; padding: 10px; width: 420px;">
      GLM-4-32B-0414
    </td>
    <td style="text-align: center; font-size: 16px; font-weight: bold; padding: 10px; width: 420px;">
      GLM-4-32B-0414
    </td>
  </tr>
  <tr>
    <td style="vertical-align: top; padding: 10px; width: 420px;">
      <img src="https://github.com/user-attachments/assets/9407e4c1-1876-4ab5-838c-839836fb418a"/>
      <div style="margin-top: 10px; font-size: 14px; color: #333; width: 400px;">
          Create a misty Jiangnan scene using SVG. (Prompt translated from Chinese)
      </div>
    </td>
    <td style="vertical-align: top; padding: 10px; width: 420px;">
      <img src="https://github.com/user-attachments/assets/bcce8c5a-cedf-45c8-b666-ddb023d5b49c"/>
      <div style="margin-top: 10px; font-size: 14px; color: #333; width: 400px;"> Use SVG to illustrate the training process of an LLM. (Prompt translated from Chinese) </div>
    </td>
  </tr>
</table>


### Search-Based Writing

For search-based writing tasks, we use the following system prompt to have the model respond based on search results:

```
请根据所给搜索返回结果对用户问题进行作答。

## 注意
1. 充分利用和整理收集到的信息，而不是简单的复制粘贴，生成符合用户要求且有深度的专业答案。
2. 所提供信息充分的情况下，你的回答需尽可能延长，从用户意图角度出发，提供具有足够信息量和多角度的回复。
3. 另外，并非所有的搜索结果都与用户问题密切相关，请仔细的甄别、筛选和利用。
4. 客观类问答的答案通常非常简短，你可以适当补充一到两句相关信息，以丰富内容。
5. 请确保你的回复格式美观、可读性强。对于多实体对比或列举，善用列表格式来帮助用户更好的理解信息。
6. 除非用户要求，否则你回答的语言请于用户提问语言保持一致。
7. 在适当情况下在句子末尾使用例如:【0†source】的格式引用搜索结果。
```

When using, you can obtain search results through methods such as `RAG` or `WebSearch`, and wrap them in `observation`, for example:

```json
[
    {
        "role": "user",
        "content": "Explore the common characteristics of children's literature, with a focus on its narrative techniques and thematic tendencies. This includes narrative techniques: common approaches in children's literature such as first-person, third-person, omniscient narrator, and interactive narration, and their influence on young readers. It also includes thematic tendencies: recurring themes in children's literature such as growth, adventure, friendship, and family, with an analysis of how these themes impact children's cognitive and emotional development. Additionally, other universal features such as the use of personification, repetitive language, symbolism and metaphor, and educational value should be considered. Please provide a detailed analytical report based on academic research, classic examples of children's literature, and expert opinions."
    },
    {
        "role": "observation",
        "content": "【{id}†{title}†{url}】\n{content}"
    },
    ...
]
```
For the above prompt, we use an internal or external search model to obtain the search results. Using the format shown above, we can generate the following analysis report:

<div style="height: 400px; width: 100%; overflow: auto; border: 5px solid #ddd; padding: 20px;">

# Analysis Report on Common Characteristics of Children's Literature

Children's literature, as a literary genre specifically created for children, possesses unique artistic features and expressive techniques. This report will comprehensively analyze the common characteristics of children's literature from three dimensions: narrative methods, thematic tendencies, and other universal features, integrating academic research, classic examples of children's literature, and expert viewpoints.

## Narrative Methods and Their Impact on Child Readers

The narrative methods of children's literature exhibit diverse characteristics, including first-person, third-person, narration, and interactive storytelling. These different narrative perspectives and methods profoundly influence children's reading experiences and cognitive development.

**First-person narration** is common in children's literature, unfolding the story from the perspective of a child protagonist. This narrative style bridges the gap between the reader and the story, making it easier for child readers to identify with the characters and develop emotional resonance. For example, many coming-of-age novels use first-person narration, allowing the protagonist to directly recount their experiences and feelings, making readers feel as though they are experiencing the joys and sorrows of growth alongside the protagonist. This narrative perspective lends the work a strong sense of authenticity and intimacy, helping to cultivate children's empathy【1†source】.

**Third-person narration** offers a broader perspective, allowing the author to flexibly switch between different characters' viewpoints and present richer layers of the story. In children's literature, third-person omniscient narration enables the author to control the narrative pace, revealing or concealing information as needed to guide children's attention. At the same time, third-person narration facilitates direct dialogue between the author and the reader, conveying values or explaining complex concepts through narration. This narrative method positively influences children's macro-thinking and comprehensive understanding【1†source】.

**Narration (authorial intrusion)** is a unique narrative technique in children's literature, where the author directly appears as the "storyteller," explaining the background, commenting on characters, or posing questions to the reader. This technique is particularly common in classic fairy tales, such as the opening lines of *Andersen's Fairy Tales*: "Once, there was a child..." Narration helps children understand the story's context, fills cognitive gaps, and conveys the author's educational intent. Research shows that appropriate authorial intrusion aids children in grasping the story's structure and improving reading comprehension【5†source】.

**Interactive storytelling** is a new trend in contemporary children's literature, especially prominent in the digital media era. Interactive storytelling breaks the traditional unidirectional author-reader relationship, encouraging child readers to participate in the story's creation, such as by choosing plot directions, character dialogues, or endings. This participatory reading enhances children's sense of agency and fosters decision-making skills and creative thinking. For example, some children's reading apps incorporate interactive elements, allowing children to influence the story's development through clicks, drag-and-drop actions, and other operations, thereby gaining a stronger sense of immersion and achievement【6†source】. Interactive storytelling transforms children from passive information recipients into active meaning-makers, uniquely contributing to the development of their subjectivity.

*Table: Common Narrative Methods in Children's Literature and Their Effects*

| **Narrative Method** | **Characteristics** | **Impact on Child Readers** | **Classic Examples** |
|----------------------|--------------------|----------------------------|---------------------|
| **First-Person** | Told from the child protagonist's perspective | Enhances immersion, fosters empathy | *Charlotte's Web*, *The Straw House* |
| **Third-Person** | Omniscient or limited perspective | Expands horizons, develops comprehensive understanding | *Harry Potter* series |
| **Narration** | Direct authorial intrusion into the narrative | Aids comprehension, conveys values | *Andersen's Fairy Tales* |
| **Interactive** | Encourages reader participation in creation | Cultivates agency and creative thinking | Children's interactive reading apps |

Notably, the narrative methods of children's literature are often closely intertwined with the **childhood perspective**. The childhood perspective does not necessarily mean the narrator must be a child but refers to the work's ability to describe the world to the greatest extent from a child's heart, expressing their inner psychology and external circumstances【2†source】. Through the childhood perspective, readers can embark on a spiritual journey with a child's mindset, a narrative strategy that creates a strong sense of realism, allowing child readers to achieve emotional identification and cognitive resonance during the reading process【1†source】. The use of the childhood perspective gives the work's language a perceptual and naive quality, often with a prose-like and spatial structure, artistic features that align with children's cognitive characteristics and aid their acceptance and understanding【2†source】.

## Thematic Tendencies and Their Impact on Children's Cognitive and Emotional Development

The thematic choices in children's literature exhibit distinct tendencies, with common themes including growth, adventure, friendship, and family. These themes not only form the core content of children's literature but also subtly influence children's cognitive development and emotional shaping.

**The theme of growth** is one of the central motifs in children's literature. Growth narratives are regarded as the artistic lifeblood of children's literature, focusing on depicting the pivotal moments of rapid psychological development in children, particularly the awakening and establishment of self-awareness【3†source】. Growth literature typically includes three elements: an artistic portrayal of the self-awareness construction process in growing adolescents, a developmental story with logical propulsion, and the presentation of the protagonist's spiritual trials and quest for direction【3†source】. By reading growth-themed works, child readers can indirectly experience the confusion and breakthroughs of growing up and understand the formation of self-identity. Classics such as Astrid Lindgren's *Pippi Longstocking* and Cao Wenxuan's *The Straw House* vividly depict children's psychological growth trajectories in specific environments. Research indicates that growth-themed literary works help children build a positive self-concept and develop the courage and resilience to face challenges, positively contributing to their psychological development【9†source】.

**The theme of adventure** holds an important place in children's literature, satisfying children's curiosity about exploring the unknown. Adventure stories often feature unusual settings and unknown challenges, with the protagonist growing through overcoming difficulties. Classics like *Robinson Crusoe* and *The Adventures of Tom Sawyer* attract child readers with thrilling plots while conveying the importance of qualities such as courage, wisdom, and perseverance. The impact of adventure themes on children's cognitive development mainly lies in expanding their imaginative space and fostering problem-solving skills. In adventure stories, children must analyze situations, make plans, and respond to unexpected events alongside the protagonist, a process that exercises their logical thinking and adaptability【14†source】. At the same time, the unfamiliar environments and novel experiences in adventure stories stimulate children's curiosity and desire to learn, laying the foundation for cultivating an exploratory spirit. As experts point out, excellent children's literature should be grounded in reality, rich in depth, and generate significant inspiration and感染力, guiding children to comprehensively understand the world【14†source】.

**The theme of friendship** is equally prevalent in children's literature, reflecting children's emphasis on peer relationships. Friendship and love are regarded as humanity's most precious qualities, often depicted in children's literature as beacons in the night, guiding children toward the future【9†source】. Friendship stories typically revolve around interactions between children, portraying positive behaviors such as sharing, cooperation, and understanding. Examples include the genuine friendships among the children at Tomoe Gakuen in *Totto-Chan: The Little Girl at the Window* and the promise and mutual aid between Wilbur and Charlotte in *Charlotte's Web*. These stories help child readers recognize the value of friendship and learn how to build and maintain interpersonal relationships. Research shows that children need peer support during their growth, as friends provide crucial emotional anchors, offering the greatest emotional support and comfort in unfamiliar environments【16†source】. By reading friendship-themed works, children can learn social skills, develop empathy, and cultivate a spirit of cooperation, qualities essential for their social development【17†source】.

**The theme of family** is an indispensable subject in children's literature, depicting the emotional bonds and interaction patterns among family members. As the primary setting for children's earliest socialization, the family atmosphere and parenting styles profoundly impact children's mental health【10†source】. Family stories in children's literature often focus on parent-child relationships, sibling bonds, and other dynamics, such as Alice's relationship with her sister in *Alice's Adventures in Wonderland* and the Little Prince's interactions with the rose in *The Little Prince*. These stories help children understand the responsibilities and expectations of family roles and learn to handle conflicts within the family. Research indicates that a positive family atmosphere and parental support promote the development of children's positive psychological traits, while adverse family environments and parenting behaviors negatively affect their mental health【10†source】【11†source】. By reading family-themed works, children can gain emotional support, learn skills for managing family relationships, and establish healthy family values.

*Table: Common Themes in Children's Literature and Their Impact on Child Development*

| **Theme Type** | **Content Representation** | **Impact on Cognitive Development** | **Impact on Emotional Development** | **Classic Examples** |
|---------------|---------------------------|-------------------------------------|-------------------------------------|---------------------|
| **Growth** | Awakening of self-awareness, psychological trials and breakthroughs | Establishes self-concept, fosters problem-solving skills | Shapes positive self-identity, enhances psychological resilience | *The Straw House*, *Pippi Longstocking* |
| **Adventure** | Exploring the unknown, overcoming challenges | Expands imaginative space, exercises logical thinking | Cultivates courage and perseverance | *Robinson Crusoe*, *The Adventures of Tom Sawyer* |
| **Friendship** | Peer interactions, mutual aid and cooperation | Learns social skills, understands interpersonal dynamics | Develops empathy, builds a sense of belonging | *Charlotte's Web*, *Totto-Chan: The Little Girl at the Window* |
| **Family** | Parent-child relationships, sibling bonds | Understands social roles, learns communication skills | Gains emotional support, establishes secure attachments | *Alice's Adventures in Wonderland*, *The Little Prince* |

Regarding thematic choices, children's literature researcher Zhu Ziqiang proposed the famous "Three Major Motifs" theory, categorizing children's literary works into "the motif of love," "the motif of the mischievous child," and "the motif of nature"【8†source】. The motif of love focuses on emotional connections between children and adults or peers; the motif of the mischievous child portrays children's free-spirited nature; and the motif of nature emphasizes the harmonious relationship between children and the natural environment. These three motifs reflect the richness of the children's world from different angles, providing diverse emotional experiences and cognitive frameworks for children. Notably, these themes do not exist in isolation; outstanding works often organically integrate multiple themes. For example, the *Harry Potter* series incorporates growth, friendship, adventure, and family elements, presenting child readers with a multidimensional spiritual world.

## Other Universal Features and Their Artistic Expression

In addition to narrative methods and thematic tendencies, children's literature exhibits a series of universal artistic features, including anthropomorphism, repetitive language, symbolism and metaphor, and educational significance. These features collectively constitute the unique aesthetic style of children's literature, subtly influencing children's cognitive development and aesthetic cultivation.

**Anthropomorphism** is one of the most distinctive artistic features of children's literature. In children's literary works, animals, plants, and even inanimate objects are often endowed with human thoughts, emotions, and behaviors, greatly enhancing the story's fun and imagination. Research shows that anthropomorphism is a frequently used technique by children's literature creators to attribute human characteristics to animals, enabling them to possess perception and communication abilities【19†source】. Through anthropomorphism, children can more easily understand abstract concepts and moral principles, as anthropomorphic characters translate complex ideas into familiar emotional and behavioral patterns. For example, in scientific fairy tales, anthropomorphic characters can help explain scientific principles, making abstract concepts tangible【18†source】. Anthropomorphism not only enriches the narrative techniques of children's literature but also provides children with a unique perspective for understanding the relationship between humans and nature. It is worth noting that excessive anthropomorphism may affect children's accurate understanding of the animal world, so modern children's literature pays more attention to balancing the natural attributes of characters with human characteristics when employing anthropomorphic techniques【19†source】.

**Repetitive language** is extremely common in children's literature, a linguistic feature rooted in oral traditions originally intended to aid memory and dissemination【20†source】. In children's literature, the repetitive use of words, phrases, or sentences serves multiple functions: constructing the story's framework, emphasizing key information, creating rhythm and musicality, and training children's vocabulary skills. For example, in *The Very Hungry Caterpillar*, the author repeatedly uses phrases like "On Monday, he ate one apple. On Tuesday, he ate two pears..." This not only builds the story's structure but also helps children learn numbers and days of the week. Repetitive structures also aid children in developing an awareness of language patterns during the early stages of language acquisition, fostering a sense of language and memory skills【21†source】. Research indicates that repetitive language in children's literature promotes children's language acquisition, helping them master vocabulary and syntactic rules. At the same time, this linguistic feature enhances the story's participatory nature, as children can often join in reciting the repetitive parts, gaining a sense of achievement.

**Symbolism and metaphor** are common expressive techniques in children's literature, conveying abstract meanings through concrete imagery. Symbolism uses specific objects to represent abstract concepts or emotions, while metaphor connects two different things through comparison, creating new meanings. In children's literature, symbolism and metaphor are usually presented in a simple and clear manner, avoiding overly complex interpretations. For example, the character configurations and metaphorical connotations in *The Wizard of Oz* are thought-provoking, as these characters not only breathe life into the story but also convey profound life philosophies through their symbolic meanings【24†source】. Symbolism and metaphor in children's literature are often related to themes such as growth, friendship, and courage, helping children understand abstract concepts through concrete and figurative expressions. Research shows that appropriate metaphors can promote children's cognitive development, stimulating their imagination and creativity【23†source】. As children grow older, their ability to understand symbolism and metaphor gradually improves, providing children's literature with multi-layered meaning spaces.

**Educational significance** is an indispensable component of children's literature, which inherently carries the gene of children's education【22†source】. Excellent children's literary works simultaneously possess entertainment and educational functions, not only helping children understand the objective world, enrich their inner emotions, and acquire life wisdom but also cultivating their perception, aesthetic sensibility, thinking skills, and creativity【15†source】. Educational significance in children's literature is often not directly presented through preaching but naturally revealed through the storyline and characters' fates. For example, many classic fairy tales convey the importance of qualities such as bravery and honesty through the protagonist's adventurous experiences, while popular science books introduces scientific knowledge through interesting plots and characters. Experts point out that children's literature writers should shoulder the importantence of education, incorporating care for children's mental growth into their works【22†source】. It is worth noting that the educational significance of children's literature should respect children's receptive abilities, avoiding excessive preaching or moral indoctrination, and instead naturally influencing children's values and behaviors through artistic appeal.

**Storytelling** is the most basic and essential feature of children's literature. Children's perceptual, imagery-driven, and novelty-seeking cognitive characteristics and receptive psychology further determine that "storytelling" is an indispensable ontological feature of children's literature【25†source】. Engaging plots are the most crucial aspect of children's literary works because, compared to adults, children's understanding of things relies mainly on intuition, and plots play a key role in guiding children's comprehension of stories【26†source】. The storytelling quality of children's literature is reflected in multiple aspects: clear cause-and-effect relationships, Compact narrative rhythm and satisfying endings. These elements work together to immerse children in the story world, providing emotional satisfaction and cognitive inspiration. As researchers have noted, plots must be performed by specific characters in specific situations to convey individual experiences in unique space-time environments【7†source】. In children's literature, storytelling is not merely an artistic technique but a bridge connecting children to the world. Through stories, children can safely experience various life scenarios and learn methods for challenges.

In terms of **language features**, children's literature typically adopts a concise, clear, and vivid language style, avoiding complex sentence structures and abstract vocabulary. This linguistic characteristic aligns with children's cognitive development levels, facilitating their understanding and acceptance. At the same time, the language of children's literature is often rich in rhythm and musicality, enhancing readability and memorability through techniques such as rhyming and repetition. For example, Michael Rosen's children's literary works extensively employ repetitive structures and rhymes, a language usage that helps children develop an awareness of language patterns during the early stages of language acquisition【21†source】. The language of children's literature also often includes rich sensory descriptions and emotional expressions, stimulating children's imagination through concrete and tangible imagery. Scholar Jay Davis's research shows that the interactive use of language in children's literature can influence children's language habits and promote their language development【21†source】.

In summary, these universal features of children's literature collectively constitute its unique artistic charm and educational value. Anthropomorphism and symbolism expand children's imaginative spaces, repetitive language and storytelling promote language acquisition and cognitive development, and the natural integration of educational significance achieves the artistic effect of "teaching through entertainment." These features do not exist in isolation but are interwoven and organically unified, collectively serving the comprehensive development of child readers.

## Conclusion

Through a systematic analysis of the narrative methods, thematic tendencies, and other universal features of children's literature, we can draw the following conclusions: As a special literary genre, the creation and reception of children's literature follow unique rules. In terms of narrative methods, children's literature flexibly employs various techniques such as first-person, third-person, narration, and interactive storytelling to adapt to children's cognitive characteristics and receptive psychology. Among these, the use of the childhood perspective is particularly important, as it enhances the work's sense of realism and intimacy, enabling child readers to develop emotional resonance【1†source】【2†source】. In terms of thematic choices, growth, adventure, friendship, and family constitute the main content of children's literature. These themes not only satisfy children's curiosity and desire to explore but also subtly influence their cognitive development and emotional shaping【3†source】【9†source】. Other universal features such as anthropomorphism, repetitive language, symbolism, and educational significance collectively form the unique artistic style and educational value of children's literature【18†source】【20†source】【24†source】.

These characteristics of children's literature do not exist in isolation but are interconnected and organically unified. For example, adventure themes are often combined with third-person omniscient narration to attract child readers through compact plots and vivid descriptions; friendship themes frequently employ first-person narration to enhance emotional resonance; and anthropomorphism is commonly found in nature-themed works, helping children understand the relationship between humans and nature. These features collectively serve the comprehensive development of child readers, meeting their entertainment needs while promoting their cognitive growth and emotional maturity.

From an academic research perspective, children's literature studies should emphasize the application of narrative theory, as narrative theory focuses more on the "how" of storytelling—narrative form—which aligns closely with the research focus of children's literature【0†source】. At the same time, cognitive research methods provide new perspectives for children's literature studies. By combining cognitive science with literary theory, we can gain a deeper understanding of how children's literature influences children's thinking and cognitive development【4†source】. Future research should continue to explore the application of these theoretical methods in children's literature studies while paying attention to the intersection and integration of children's literature with emerging fields such as digital media and interdisciplinary education.

From a creative practice perspective, children's literature writers should fully grasp children's cognitive characteristics and emotional needs, incorporating growth Care and educational wisdom into their work As experts have pointed out, excellent children's literary works should be grounded in reality, rich in depth, and generate significant infection and infectivity, guiding children to comprehensively understand the world and correctly recognize themselves and society【14†source】. At the same time, children's literature Creativity should keep pace with the times, addressing new problems and challenges faced by contemporary children, such as media literacy in the digital age and identity formation in multicultural contexts, to provide targeted spiritual nourishment for children.

From an educational application perspective, children's literature should fully leverage its unique role in children's mental growth. Through carefully designed reading activities, teachers and parents can help children deeply understand the themes and meanings in works, guiding them to connect reading experiences with real life. Research shows that children's literature plays an increasingly important role in language education, the construction of a reading society, and children's mental growth【22†source】. Therefore, children's literature should be incorporated as an important component of school and family education, promoting children's cognitive development and emotional maturity through activities such as reading sharing, role-playing, and creative writing.

In summary, as a unique art form and educational medium, the common characteristics of children's literature constitute an organic whole, collectively serving the comprehensive development of child readers. By deeply understanding these features and their mechanisms of influence, we can better create, research, and apply children's literature, providing high-quality spiritual nourishment for children's healthy growth. Future children's literature research should continue to deepen theoretical exploration, expand research methods, and strengthen interdisciplinary collaboration to address the ever-changing needs of children and the challenges of the times, promoting the continuous development of children's literature.

</div>
    
### Function Call

GLM-4-32B-0414 supports calling external tools in JSON format. This can be done via HuggingFace Transformers, vLLM, or sgLang.  
The message format for tool calling is as follows:
```json=
{
    "role": "asssitant", 
    "metadata": function_name, 
    "content": json.dumps(call_arguments, ensure_ascii=False)
}
```

The message format for tool execution results is as follows:

```json=
{
    "role": "observation", 
    "content": json.dumps(tool_response, ensure_ascii=False) if not isinstance(tool_response, str) else tool_response
}
```

The following example demonstrates the process of GLM-4-9B-0414 calling a tool and generating a final response using HuggingFace Transformers.

```python
import json
import re
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "THUDM/GLM-4-9B-0414"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

def is_function_call(single_message):
    """Determine whether the current system message is a function call."""
    pattern = re.compile(r'([^\n`]*?)\n({.*?})(?=\w*\n|$)', re.DOTALL)
    matches = pattern.findall(single_message)
    if not matches:
        return False

    func_name, args_str = matches[0]
    func_name = func_name.strip()
    try:
        parsed_args = json.loads(args_str)
    except json.JSONDecodeError:
        try:
            parsed_args = ast.literal_eval(args_str)
        except:
            return False
    
    return {"name": func_name, "arguments": parsed_args}

def realtime_aqi(city):
    """Weather Query Tool"""
    if '北京' in city.lower():
        return json.dumps({'city': '北京', 'aqi': '10', 'unit': 'celsius'}, ensure_ascii=False)
    elif '上海' in city.lower():
        return json.dumps({'city': '上海', 'aqi': '72', 'unit': 'fahrenheit'}, ensure_ascii=False)
    else:
        return json.dumps({'city': city, 'aqi': 'unknown'}, ensure_ascii=False)

def build_system_prompt(tools):
    """Construct system prompt based on the list of available tools."""
    if tools is None:
        tools = []
    value = "# 可用工具"
    contents = []
    for tool in tools:
        content = f"\n\n## {tool['function']['name']}\n\n{json.dumps(tool['function'], ensure_ascii=False, indent=4)}"
        content += "\n在调用上述函数时，请使用 Json 格式表示调用的参数。"
        contents.append(content)
    value += "".join(contents)
    return value

tools = [
  {
    "type": "function", 
    "function": {
      "name": "realtime_aqi",
      "description": "天气预报。获取实时空气质量。当前空气质量，PM2.5，PM10信息",
      "parameters": {
          "type": "object",
          "properties": {
              "city": {
                  "description": "城市名"
              }
          },
          "required": [
              "city"
          ]
      }
	}
  }
]

system_prompt = build_system_prompt(tools)

message = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "北京和上海今天的天气情况"}
]
print(f"User Message: {message[-1]['content']}")

while True:
    inputs = tokenizer.apply_chat_template(
        message,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    generate_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 1024,
        "do_sample": True,
    }
    out = model.generate(**generate_kwargs)
    generate_resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:-1], skip_special_tokens=False)
    stop_sequence = tokenizer.decode(out[0][-1:], skip_speical_tokens=False)
    if stop_sequence == "<|user|>":
        print(f"Assistant Response: {generate_resp.strip()}")
        break

    function_calls = []
    for m in generate_resp.split("<|assistant|>"):
        fc_decode = is_function_call(m.strip())
        if fc_decode:
            message.append({"role": "assistant", "metadata": fc_decode['name'], "content": json.dumps(fc_decode['arguments'], ensure_ascii=False)})
            print(f"Function Call: {fc_decode}")
            function_calls.append(fc_decode)
        else:
            message.append({"role": "assistant", "content": m})
            print(f"Assistant Response: {m.strip()}")
    
    for fc in function_calls:
        function_response = realtime_aqi(
            city=fc["arguments"]["city"],
        )
        print(f"Function Response: {function_response}")
        message.append({"role": "observation", "content": function_response})
```

## Evaluation Results

<div style="text-align: center;">
  <img src="https://raw.githubusercontent.com/THUDM/GLM-4/refs/heads/main/resources/Bench-32B.png" style="width: 80%;" />
</div>

### GLM-4-0414 Series

| 模型             | IFEval | BFCL-v3 (Overall) | BFCL-v3 (MultiTurn) | TAU-Bench (Retail) | TAU-Bench (Airline) | SimpleQA | HotpotQA |
| ---------------- | ------ | ----------------- | ------------------- | ------------------ | ------------------- | -------- | -------- |
| Qwen2.5-Max      | 85.6   | 50.9              | 30.5                | 58.3               | 22.0                | 79.0     | 52.8     |
| GPT-4o-1120      | 81.9   | 69.6              | 41.0                | 62.8               | 46.0                | 82.8     | 63.9     |
| DeepSeek-V3-0324 | 83.4   | 66.2              | 35.8                | 60.7               | 32.4                | 82.6     | 54.6     |
| DeepSeek-R1      | 84.3   | 57.5              | 12.4                | 33.0               | 37.3                | 83.9     | 63.1     |
| GLM-4-32B-0414   | 87.6   | 69.6              | 41.5                | 68.7               | 51.2                | 88.1     | 63.8     |

> For `SimpleQA` and `HotpotQA`, we sampled nearly 500 test cases from each test set, provided all models with basic `search` and `click` tools, ensured other settings remained consistent, and averaged the results over 3 runs.

| Model  | Framework  | [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)  | [SWE-bench Verified mini](https://github.com/mariushobbhahn/SWEBench-verified-mini) |
|---|---|---|---|
| GLM-4-32B-0414  | Moatless<sup>[1]</sup> | 33.8 | 38.0 |
| GLM-4-32B-0414  | Agentless<sup>[2]</sup>  | 30.7 | 34.0 |
| GLM-4-32B-0414  | OpenHands<sup>[3]</sup> | 27.2  | 28.0  |

[1] [Moatless v0.0.3](https://github.com/aorwall/moatless-tools) used the following parameters: `response_format="react", thoughts_in_action=False, max_interations=30`. No retries on failed trajectories; other settings are default.

[2] [Agentless v1.5.0](https://github.com/OpenAutoCoder/Agentless) used [BGE](https://github.com/FlagOpen/FlagEmbedding/blob/master/README.md) as the embedding model and [FAISS](https://github.com/facebookresearch/faiss) for similarity search. To speed up patch verification while maintaining performance, the timeout for running a single instance was changed from the default 300s to 180s.

[3] [OpenHands v0.29.1](https://github.com/All-Hands-AI/OpenHands/tree/main) did not use YaRN context extension but limited runs to a maximum of 60 iterations and summarized the history to prevent exceeding the 32K context limit. Summarization was configured as `llm_config="condenser", keep_first=1, max_size=32`. No retries on failed trajectories.
