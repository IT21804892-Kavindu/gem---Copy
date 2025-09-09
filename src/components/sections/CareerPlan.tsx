import React from 'react';
import { SectionHeading } from '../ui/SectionHeading';
import { Calendar, Target, TrendingUp, Milestone, Trophy } from 'lucide-react';

export const CareerPlan: React.FC = () => {
  const careerGoals = [
    {
      timeframe: "Short-term Goals (1 year)",
      icon: <Calendar className="w-6 h-6 text-blue-600" />,
      goals: [
        "Master full-stack development with MERN stack and expand to other frameworks",
        "Complete advanced cloud certification (AWS or Azure)",
        "Contribute to at least 2 open-source projects",
        "Improve technical writing skills through blog posts and documentation"
      ],
      actions: [
        "Enroll in specialized courses for advanced JavaScript and React concepts",
        "Schedule weekly time for open-source contributions",
        "Create a personal blog to document learning journey",
        "Participate in coding challenges and hackathons"
      ]
    },
    {
      timeframe: "Medium-term Goals (2-3 years)",
      icon: <Target className="w-6 h-6 text-blue-600" />,
      goals: [
        "Transition to a senior software engineer role",
        "Develop expertise in system architecture and design patterns",
        "Lead development of at least one major project",
        "Mentor junior developers and improve leadership skills"
      ],
      actions: [
        "Seek opportunities to lead smaller projects within current role",
        "Study advanced system design and architecture patterns",
        "Participate in mentorship programs within the organization",
        "Attend leadership workshops and management training"
      ]
    },
    {
      timeframe: "Long-term Goals (4-5 years)",
      icon: <TrendingUp className="w-6 h-6 text-blue-600" />,
      goals: [
        "Move into a technical lead or engineering manager position",
        "Develop specialized expertise in AI/ML or cybersecurity",
        "Contribute to technical standards or publications in the field",
        "Create and lead a technical team on innovative projects"
      ],
      actions: [
        "Pursue advanced degree or specialized certification in chosen area",
        "Attend industry conferences and networking events",
        "Seek speaking opportunities at technical events",
        "Develop mentorship relationships with industry leaders"
      ]
    }
  ];

  const skills = [
    {
      category: "Technical Skills to Acquire",
      icon: <Milestone className="w-6 h-6 text-teal-600" />,
      items: [
        "Advanced cloud architecture and DevOps practices",
        "Machine learning and data analysis",
        "Mobile development with React Native",
        "Microservices architecture and implementation",
        "Blockchain technology fundamentals"
      ]
    },
    {
      category: "Professional Development Targets",
      icon: <Trophy className="w-6 h-6 text-teal-600" />,
      items: [
        "Advanced project management certification",
        "Technical leadership training",
        "Public speaking and presentation skills",
        "Business analysis and requirements gathering",
        "Team building and conflict resolution strategies"
      ]
    }
  ];

  return (
    <section id="career-plan" className="py-20 bg-gradient-to-br from-gray-50 to-white">
      <div className="container mx-auto px-4 md:px-6">
        <SectionHeading 
          title="Career Development Plan" 
          subtitle="A structured 3-5 year career roadmap to guide my professional growth" 
        />
        
        <div className="space-y-16">
          {careerGoals.map((timeframe, index) => (
            <div key={index} className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
              <div className="flex items-center mb-6">
                <div className="bg-blue-50 p-3 rounded-full mr-4">
                  {timeframe.icon}
                </div>
                <h3 className="text-2xl font-bold text-gray-800">{timeframe.timeframe}</h3>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                  <h4 className="text-lg font-semibold text-blue-600 mb-4">Goals</h4>
                  <ul className="space-y-3">
                    {timeframe.goals.map((goal, idx) => (
                      <li key={idx} className="flex items-start">
                        <span className="inline-block w-5 h-5 bg-blue-600 rounded-full mr-3 mt-1 flex-shrink-0"></span>
                        <span className="text-gray-700">{goal}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div>
                  <h4 className="text-lg font-semibold text-blue-600 mb-4">Action Steps</h4>
                  <ul className="space-y-3">
                    {timeframe.actions.map((action, idx) => (
                      <li key={idx} className="flex items-start">
                        <span className="inline-block w-5 h-5 border-2 border-blue-600 rounded-full mr-3 mt-1 flex-shrink-0"></span>
                        <span className="text-gray-700">{action}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          ))}
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-12">
            {skills.map((skillSet, index) => (
              <div key={index} className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
                <div className="flex items-center mb-6">
                  <div className="bg-teal-50 p-3 rounded-full mr-4">
                    {skillSet.icon}
                  </div>
                  <h3 className="text-xl font-bold text-gray-800">{skillSet.category}</h3>
                </div>
                
                <ul className="space-y-3">
                  {skillSet.items.map((item, idx) => (
                    <li key={idx} className="flex items-start">
                      <span className="inline-block w-4 h-4 bg-teal-600 rounded-sm mr-3 mt-1 flex-shrink-0"></span>
                      <span className="text-gray-700">{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};