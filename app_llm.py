import streamlit as st
from gpt4alltest import get_summary_from_llm_gpt4all
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

st.title("Application: Talking with LLM")
text_area_placeholder = st.empty()
texto_input = text_area_placeholder.text_area("Give me a text to summarize", value='', height=200)

if st.button('Process'):
    text_area_placeholder.empty()
    st.subheader("Your text:")
    st.markdown(f'**Text:** {texto_input}')
    logging.info(f"Start process with llm")
    text_summarize = get_summary_from_llm_gpt4all(texto_input, model_name="llama-2-7b-chat")
    logging.info(f"End process with llm")
    st.subheader("Your summary:")
    st.write(text_summarize)

text = """The rapid advancement of technology in the 21st century has transformed the way we live, work, 
and communicate. From the proliferation of smartphones to the rise of artificial intelligence, our world is 
constantly evolving. This text will explore the impact of technology on various aspects of our lives.

In the realm of communication, the internet and social media have revolutionized how we connect with one another. 
With platforms like Facebook, Twitter, and Instagram, we can instantly share our thoughts and experiences with 
friends and followers from around the globe. Online communication has also become essential in business, with email 
and video conferencing enabling companies to collaborate across borders.

The workplace itself has undergone significant changes due to technology. Automation and digitalization have reshaped 
industries, leading to increased efficiency and productivity. However, concerns about job displacement and the need 
for retraining workers have emerged as important considerations.

Education is another domain where technology has left a lasting impact. Online courses and e-learning platforms have 
made education more accessible, allowing students to learn at their own pace. Furthermore, virtual reality (VR) and 
augmented reality (AR) technologies are transforming the way we learn and experience the world, offering immersive 
educational opportunities.

The healthcare industry has benefited from technological advancements as well. From telemedicine to robotic 
surgeries, technology is improving patient care and expanding access to medical services. With the growth of wearable 
devices and health apps, individuals can monitor their well-being and take proactive measures to maintain a healthy 
lifestyle.

In the realm of entertainment, streaming services have replaced traditional television and movie theaters. Platforms 
like Netflix and Disney+ offer a vast library of content at our fingertips, while video games have become a dominant 
form of entertainment, with esports gaining global popularity.

The automotive industry is in the midst of a revolution, with electric and autonomous vehicles on the horizon. These 
innovations promise to reduce greenhouse gas emissions and make transportation safer and more efficient. 
Additionally, smart cities are integrating technology to enhance urban living, with initiatives such as 
energy-efficient buildings and connected infrastructure.

Security and privacy have become paramount concerns in our digital age. Cybersecurity measures are continuously 
evolving to protect individuals and organizations from cyber threats. Meanwhile, debates about data privacy and 
surveillance continue to shape public discourse.

As technology continues to advance, it offers immense opportunities and challenges. Finding the right balance between 
innovation and ethical considerations is crucial. With responsible use and thoughtful regulation, technology can 
continue to enhance our lives, drive economic growth, and address some of the world's most pressing issues.

The intersection of technology and society is a complex and multifaceted one. It raises questions about ethics, 
privacy, inequality, and the future of work. As we navigate this rapidly changing landscape, it's essential to 
consider the implications of our technological advancements and to strive for a future where technology benefits all 
of humanity.

One of the central themes in the ongoing technological evolution is the idea of artificial intelligence (AI). AI 
encompasses a wide range of technologies that aim to replicate or simulate human intelligence in machines. This 
includes machine learning, deep learning, natural language processing, and computer vision.

The impact of AI is already being felt in various industries. In healthcare, AI-powered systems can analyze medical 
images with a level of precision that surpasses human capabilities. These systems can assist doctors in diagnosing 
diseases, from cancer to neurological disorders, earlier and more accurately.

In the financial sector, AI is revolutionizing how we manage and invest money. Algorithmic trading systems can 
analyze vast amounts of financial data in real-time, making split-second investment decisions. Additionally, 
AI-driven robo-advisors can provide personalized investment recommendations to individuals.

The world of transportation is also experiencing a profound transformation thanks to AI. Self-driving cars and trucks 
promise to make our roads safer and more efficient. They have the potential to reduce accidents caused by human error 
and improve traffic flow in congested urban areas.

In the field of education, AI is making personalized learning a reality. Intelligent tutoring systems can adapt to 
the needs of individual students, providing tailored lessons and feedback. This can significantly enhance the 
learning experience and help students of all ages achieve their educational goals.

AI has a significant presence in the realm of virtual assistants and chatbots. Companies are increasingly integrating 
AI-driven chatbots into their websites and customer support services. These bots can answer customer inquiries, 
provide recommendations, and assist with a wide range of tasks.

The entertainment industry is also exploring the potential of AI. AI algorithms can generate music, art, 
and even entire stories. This creative application of AI offers new possibilities for artists and creators.

However, the adoption of AI also raises important ethical and societal questions. Concerns about data privacy, 
bias in algorithms, and the potential for job displacement due to automation have become subjects of intense debate.

In response to these concerns, researchers and policymakers are working to establish ethical guidelines and 
regulations for AI. Transparency, accountability, and fairness are central principles in ensuring that AI 
technologies are developed and used responsibly.

It's clear that the future of technology, including AI, is exciting and full of promise, but it also presents 
challenges that we must address. As we continue to embrace technological innovations, we must do so with a sense of 
responsibility and a commitment to ensuring that these innovations benefit society as a whole.

As AI continues to permeate various industries, its impact becomes increasingly evident. In healthcare, AI-powered 
systems have reached remarkable levels of precision, outperforming human capabilities in medical image analysis. 
These systems are assisting medical professionals in early and accurate disease diagnoses, spanning conditions from 
cancer to neurological disorders. Patients are benefiting from improved healthcare outcomes as a result.

The financial sector has undergone a fundamental transformation thanks to AI. Algorithmic trading systems analyze 
vast volumes of financial data in real-time, enabling lightning-fast investment decisions. Moreover, 
AI-driven robo-advisors have made personalized investment recommendations accessible to individuals, democratizing 
wealth management.

The world of transportation is on the cusp of a revolutionary shift due to AI. Self-driving cars and autonomous 
trucks hold the promise of safer and more efficient roadways. These technologies have the potential to reduce 
accidents stemming from human error and ameliorate traffic congestion in bustling urban areas, drastically improving 
transportation.

AI is reshaping education by making personalized learning a reality. Intelligent tutoring systems adapt to the unique 
needs of individual students, delivering tailored lessons and feedback. This not only enhances the learning 
experience but also empowers students of all ages to achieve their educational goals more effectively.

AI-driven virtual assistants and chatbots are becoming an integral part of the digital landscape. Companies are 
increasingly incorporating AI-driven chatbots into their websites and customer support services. These versatile bots 
can address customer inquiries, provide product or service recommendations, and execute a wide array of tasks, 
streamlining customer interactions and support.

The entertainment industry is also exploring the creative potential of AI. AI algorithms can generate music, art, 
and entire narratives. This creative application of AI opens up novel opportunities for artists and content creators, 
expanding the boundaries of what is achievable in the world of arts and entertainment.

However, the adoption of AI brings to the forefront important ethical and societal questions. Data privacy concerns, 
bias in AI algorithms, and the potential displacement of jobs due to automation have become subjects of intense 
debate. The responsible and ethical development and use of AI are paramount."""

