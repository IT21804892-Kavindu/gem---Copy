import React from 'react';
import { SectionHeading } from '../ui/SectionHeading';
import { BookOpen, BookMarked, GraduationCap, BrainCircuit, LightbulbIcon } from 'lucide-react';

export const Journal: React.FC = () => {
  const journalEntries = [
    {
      title: "Key Lessons Learned",
      icon: <BookOpen className="w-8 h-8 text-blue-600" />,
      content: "Throughout my Professional Practice Workshop (PPW), I've gained invaluable insights into software development methodologies. I've learned the importance of Agile practices, effective communication in team environments, and how to approach problem-solving methodically. The emphasis on clean code and maintainable architecture has significantly improved my programming practices."
    },
    {
      title: "Skills Developed",
      icon: <BrainCircuit className="w-8 h-8 text-blue-600" />,
      content: "The PPW has been instrumental in enhancing both my technical and soft skills. I've strengthened my proficiency in full-stack development, database management, and version control. Additionally, I've developed crucial project management abilities, improved my documentation practices, and honed my skills in code review and quality assurance processes."
    },
    {
      title: "Challenges Overcome",
      icon: <BookMarked className="w-8 h-8 text-blue-600" />,
      content: "During the workshop, I faced several challenges that pushed me out of my comfort zone. Debugging complex system issues, integrating different technologies, and meeting tight deadlines were particularly demanding. Through perseverance and collaboration with peers, I developed strategies to overcome these obstacles, building my resilience and problem-solving capabilities."
    },
    {
      title: "Personal Growth Insights",
      icon: <LightbulbIcon className="w-8 h-8 text-blue-600" />,
      content: "The PPW journey has contributed significantly to my personal growth. I've become more adaptable to changing requirements, more confident in presenting my ideas, and more effective in collaborating with diverse team members. I've also developed a deeper appreciation for continuous learning and the importance of seeking feedback."
    },
    {
      title: "Future Applications",
      icon: <GraduationCap className="w-8 h-8 text-blue-600" />,
      content: "The knowledge and skills acquired during the PPW will be invaluable for my future career. I plan to apply the principles of efficient coding and testing in upcoming projects, leverage my enhanced teamwork abilities, and continue refining my technical expertise. The foundation laid during this workshop will serve as a springboard for my professional development."
    }
  ];

  return (
    <section id="journal" className="py-20 bg-white">
      <div className="container mx-auto px-4 md:px-6">
        <SectionHeading 
          title="Reflective Journal" 
          subtitle="Documenting my Professional Practice Workshop learning journey" 
        />
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {journalEntries.map((entry, index) => (
            <div 
              key={index} 
              className="bg-gray-50 p-6 rounded-lg shadow-md border border-gray-200 hover:shadow-lg transition-shadow"
            >
              <div className="flex items-start mb-4">
                <div className="bg-blue-50 p-3 rounded-full mr-4">
                  {entry.icon}
                </div>
                <div>
                  <h3 className="text-xl font-bold text-gray-800 mb-2">{entry.title}</h3>
                  <p className="text-gray-600">{entry.content}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
        
        <div className="mt-12 bg-blue-50 p-6 rounded-lg border border-blue-100">
          <h3 className="text-xl font-bold text-gray-800 mb-4">Reflection Summary</h3>
          <p className="text-gray-600">
            My PPW experience has been transformative, equipping me with both technical expertise and professional skills essential for success in the software engineering field. The workshop's emphasis on practical application and collaborative learning has prepared me to tackle real-world challenges with confidence. As I continue my professional journey, I will carry forward these lessons, constantly seeking to improve and adapt in this ever-evolving industry.
          </p>
        </div>
      </div>
    </section>
  );
};