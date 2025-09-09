import React from 'react';
import { SectionHeading } from '../ui/SectionHeading';
import { Award, Calendar, ExternalLink } from 'lucide-react';

export const Certifications: React.FC = () => {
  const certifications = [
    {
      name: "AWS Certified Solutions Architect - Associate",
      organization: "Amazon Web Services",
      date: "March 2025",
      description: "Validates expertise in designing distributed systems on AWS, including deployment mechanisms, AWS-specific concepts, and best practices."
    },
    {
      name: "Professional Scrum Master I (PSM I)",
      organization: "Scrum.org",
      date: "November 2024",
      description: "Demonstrates understanding of Scrum framework, its application, and the role of Scrum Master in enabling team effectiveness."
    },
    {
      name: "MongoDB Certified Developer Associate",
      organization: "MongoDB University",
      date: "August 2024",
      description: "Validates skills in developing applications using MongoDB, including data modeling, CRUD operations, indexing, and performance optimization."
    },
    {
      name: "Microsoft Certified: Azure Developer Associate",
      organization: "Microsoft",
      date: "June 2024",
      description: "Demonstrates ability to design, build, test, and maintain cloud applications and services on Microsoft Azure."
    },
    {
      name: "Certified Kubernetes Administrator (CKA)",
      organization: "Cloud Native Computing Foundation",
      date: "February 2024",
      description: "Validates skills in installing, configuring, and managing Kubernetes clusters for container orchestration."
    },
    {
      name: "JavaScript Algorithms and Data Structures",
      organization: "freeCodeCamp",
      date: "December 2023",
      description: "Demonstrates proficiency in fundamental programming concepts, algorithms, and data structures in JavaScript."
    }
  ];

  return (
    <section id="certifications" className="py-20 bg-gradient-to-br from-blue-50 to-white">
      <div className="container mx-auto px-4 md:px-6">
        <SectionHeading 
          title="Certifications" 
          subtitle="Showcasing my recent technical and soft skill certificates" 
        />
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {certifications.map((cert, index) => (
            <div 
              key={index} 
              className="bg-white p-6 rounded-lg shadow-md border border-gray-200 hover:shadow-lg transition-shadow"
            >
              <div className="flex justify-between items-start mb-4">
                <div className="bg-blue-100 p-2 rounded-full">
                  <Award className="w-6 h-6 text-blue-600" />
                </div>
                <div className="flex items-center text-sm text-gray-600">
                  <Calendar className="w-4 h-4 mr-1" />
                  <span>{cert.date}</span>
                </div>
              </div>
              
              <h3 className="text-xl font-bold text-gray-800 mb-2">{cert.name}</h3>
              <p className="text-blue-600 font-medium mb-3">{cert.organization}</p>
              <p className="text-gray-600 mb-4">{cert.description}</p>
              
              <button className="flex items-center text-sm font-medium text-blue-600 hover:text-blue-800 transition-colors">
                View Certificate <ExternalLink className="w-4 h-4 ml-1" />
              </button>
            </div>
          ))}
        </div>
        
        <div className="mt-12 text-center">
          <p className="text-gray-600 mb-6">
            These certifications represent my commitment to continuous learning and professional development.
            I'm constantly seeking new opportunities to expand my knowledge and skills in the rapidly evolving tech landscape.
          </p>
          
          <div className="inline-flex items-center justify-center p-1 rounded-md bg-blue-100">
            <p className="text-blue-800 px-3 py-1">
              <span className="font-semibold">Upcoming:</span> Google Professional Cloud Architect (Expected: August 2025)
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};