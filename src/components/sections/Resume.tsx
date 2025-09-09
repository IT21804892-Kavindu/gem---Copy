import React from 'react';
import { SectionHeading } from '../ui/SectionHeading';
import { Briefcase, GraduationCap, Award, Code, Download } from 'lucide-react';
import cvPdf from '../images/Resume - Kavindu Rajapaksha.pdf';

export const Resume: React.FC = () => {
  const experience = [
    {
      title: "System Administrator (Part-time)",
      company: "Server Salad Cloud Services",
      location: "Sri Jayawardenepura Kotte, Sri Lanka",
      period: "Jan 2025 - Present",
      technologies: "Apache | Jenkins | AWS | Terraform | Linux | Python",
      responsibilities: [
        "Managed and optimized cloud infrastructure on AWS, including provisioning EC2 instances, configuring S3 buckets, and automating deployments using Terraform.",
        "Utilized key technologies including Apache, Jenkins, AWS, Terraform, Linux, and Python to streamline operations and automate workflows.",
        "Researched and evaluated hosting solutions (VPS, WordPress hosting, domain selling) to enhance business growth and improve service offerings."
      ]
    },
    {
      title: "Trainee Software Engineer (ERP)",
      company: "Pentagon I Runway (Pvt) Ltd",
      location: "Nugegoda, Sri Lanka",
      period: "May 2024 - Present",
      technologies: "Axpert ERP | Oracle | PostgreSQL | MySQL | DBeaver | Toad",
      responsibilities: [
        "Developed ERP solutions by analyzing client requirements and tailoring functionalities to meet business needs.",
        "Designed and optimized database schemas to improve system performance and support scalable applications.",
        "Conducted routine maintenance tasks such as backups, indexing, and performance monitoring to ensure database reliability.",
        "Extracted, transformed, and analyzed data to generate actionable insights and support business operations."
      ]
    }
  ];

  const education = [
    {
      degree: "BSc (Hons) in Information Technology",
      specialization: "Specialized in Information Technology",
      institution: "Sri Lanka Institute of Information & Technology",
      period: "2021 - Present",
      details: "CGPA up to 3rd-year - 3.17"
    },
    {
      degree: "BSc (Hons) Civil and Structural Engineering",
      institution: "Liverpool John Moores University",
      period: "2020",
      details: "Graduated with Honors, class 1"
    },
    {
      degree: "GCE Advanced Level - Physical Science",
      institution: "Maliyadeva College, Kurunegala",
      period: "2016",
      details: ""
    }
  ];

  const projects = [
    {
      title: "Web Based Garbage Collection System",
      description: "Developed a comprehensive web-based application to revolutionize the garbage collection process introducing a chatbot and a new feature called Community Swap.",
      technologies: "Bootstrap, MongoDB, ExpressJs, ReactJs, NodeJs, GitHub"
    },
    {
      title: "Paddy Farming Management System",
      description: "Developed a comprehensive web-based application to computerize and manage all the activities involving paddy farming.",
      technologies: "Bootstrap, MongoDB, ExpressJs, ReactJs, NodeJs, HTML, JavaScript, CSS, GitHub"
    },
    {
      title: "Patient Management System",
      description: "Developed a mobile application to enroll patients according to their appointments using Activity, Fragments, and Recycler View.",
      technologies: "Kotlin, Room Database, GitHub"
    },
    {
      title: "Social Media app for Fitness",
      description: "Developed a social media app for fitness enthusiasts to post their achievements, create meal plans, and interact with others.",
      technologies: "Spring Boot, MongoDB, ExpressJs, ReactJs, NodeJs, GitHub"
    }
  ];

  const skills = [
    "Linux", "Windows Server", "Jenkins", "AWS", "Docker", "Terraform", "GitHub",
    "PHP", "Java", "Python", "Angular", "MySQL", "Tailwind", "Bootstrap",
    "HTML", "JavaScript", "CSS", "MongoDB", "Express JS", "ReactJS", "NodeJS",
    "Git", "Postman", "Figma"
  ];

  const handleDownloadClick = () => {
    // Fetch the PDF as a blob and open it in a new tab
    fetch(cvPdf)
      .then(res => res.blob())
      .then(blob => {
        const fileURL = URL.createObjectURL(blob);
        window.open(fileURL, '_blank');
      })
      .catch(error => {
        console.error('Failed to open CV:', error);
     });}

  return (
    <section id="resume" className="py-20 bg-white">
      <div className="container mx-auto px-4 md:px-6">
        <SectionHeading 
          title="Curriculum Vitae" 
          subtitle="My professional experience, education, and skills" 
        />
        
        <div className="flex justify-end mb-6">
          <button 
            onClick={handleDownloadClick}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md transition-colors"
          >
            <Download size={18} /> Download CV
          </button>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            {/* Experience Section */}
            <div className="mb-12">
              <div className="flex items-center mb-6">
                <Briefcase className="w-6 h-6 text-blue-600 mr-3" />
                <h3 className="text-2xl font-bold text-gray-800">Professional Experience</h3>
              </div>
              
              <div className="space-y-8">
                {experience.map((job, index) => (
                  <div key={index} className="border-l-4 border-blue-600 pl-4 ml-3">
                    <div className="flex flex-col md:flex-row md:justify-between md:items-start mb-2">
                      <h4 className="text-xl font-bold text-gray-800">{job.title}</h4>
                      <span className="text-sm font-semibold text-blue-600 bg-blue-50 px-3 py-1 rounded-full">
                        {job.period}
                      </span>
                    </div>
                    <p className="text-gray-600 mb-2">{job.company} | {job.location}</p>
                    <p className="text-gray-700 mb-3 italic">Key Technologies: {job.technologies}</p>
                    <ul className="space-y-2">
                      {job.responsibilities.map((resp, idx) => (
                        <li key={idx} className="flex items-start">
                          <span className="inline-block w-2 h-2 bg-blue-600 rounded-full mr-3 mt-2 flex-shrink-0"></span>
                          <span className="text-gray-700">{resp}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Education Section */}
            <div className="mb-12">
              <div className="flex items-center mb-6">
                <GraduationCap className="w-6 h-6 text-blue-600 mr-3" />
                <h3 className="text-2xl font-bold text-gray-800">Education</h3>
              </div>
              
              <div className="space-y-6">
                {education.map((edu, index) => (
                  <div key={index} className="border-l-4 border-blue-600 pl-4 ml-3">
                    <div className="flex flex-col md:flex-row md:justify-between md:items-start mb-2">
                      <h4 className="text-xl font-bold text-gray-800">{edu.degree}</h4>
                      <span className="text-sm font-semibold text-blue-600 bg-blue-50 px-3 py-1 rounded-full">
                        {edu.period}
                      </span>
                    </div>
                    <p className="text-gray-600 mb-1">{edu.institution}</p>
                    {edu.specialization && (
                      <p className="text-gray-700 mb-1">{edu.specialization}</p>
                    )}
                    {edu.details && (
                      <p className="text-gray-700 italic">{edu.details}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
            
            {/* Projects Section */}
            <div>
              <div className="flex items-center mb-6">
                <Code className="w-6 h-6 text-blue-600 mr-3" />
                <h3 className="text-2xl font-bold text-gray-800">Projects</h3>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {projects.map((project, index) => (
                  <div key={index} className="bg-gray-50 p-5 rounded-lg border border-gray-200 hover:shadow-md transition-shadow">
                    <h4 className="text-lg font-bold text-gray-800 mb-2">{project.title}</h4>
                    <p className="text-gray-700 mb-3">{project.description}</p>
                    <p className="text-sm text-gray-600"><span className="font-semibold">Technologies:</span> {project.technologies}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          <div>
            {/* Skills Section */}
            <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
              <div className="flex items-center mb-6">
                <Award className="w-6 h-6 text-blue-600 mr-3" />
                <h3 className="text-2xl font-bold text-gray-800">Technical Skills</h3>
              </div>
              
              <div className="flex flex-wrap gap-2">
                {skills.map((skill, index) => (
                  <span 
                    key={index} 
                    className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>
            
            {/* Personal Information */}
            <div className="bg-gray-50 p-6 rounded-lg border border-gray-200 mt-6">
              <h3 className="text-xl font-bold text-gray-800 mb-4">Contact Information</h3>
              <ul className="space-y-3">
                <li className="flex items-center">
                  <span className="font-semibold text-gray-700 w-20">Phone:</span>
                  <a href="tel:+94768565493" className="text-blue-600 hover:underline">+94 76 8565493</a>
                </li>
                <li className="flex items-center">
                  <span className="font-semibold text-gray-700 w-20">Email:</span>
                  <a href="mailto:kavindu.kpr@gmail.com" className="text-blue-600 hover:underline">kavindu.kpr@gmail.com</a>
                </li>
                <li className="flex items-center">
                  <span className="font-semibold text-gray-700 w-20">Location:</span>
                  <span className="text-gray-700">Melsiripura, Sri Lanka</span>
                </li>
                <li className="flex items-center">
                  <span className="font-semibold text-gray-700 w-20">LinkedIn:</span>
                  <a href="https://www.linkedin.com/in/kavindu-rajapaksha-in/" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline truncate">kavindu-rajapaksha-in</a>
                </li>
              </ul>
            </div>
            
            {/* Referees */}
            <div className="bg-gray-50 p-6 rounded-lg border border-gray-200 mt-6">
              <h3 className="text-xl font-bold text-gray-800 mb-4">Referees</h3>
              <div className="space-y-5">
                <div>
                  <h4 className="font-bold text-gray-800">Namalie Walgampaya</h4>
                  <p className="text-gray-600 text-sm">Senior Lecturer</p>
                  <p className="text-gray-600 text-sm">Sri Lanka Institute of Information & Technology</p>
                  <div className="mt-1">
                    <p className="text-sm">
                      <span className="font-semibold">Phone:</span> +94 773850140
                    </p>
                    <p className="text-sm">
                      <span className="font-semibold">Email:</span> namalie.w@sliit.lk
                    </p>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-bold text-gray-800">Chathuranga Herath</h4>
                  <p className="text-gray-600 text-sm">DevSecOps Engineer</p>
                  <p className="text-gray-600 text-sm">Wavenet International (pvt) Ltd</p>
                  <div className="mt-1">
                    <p className="text-sm">
                      <span className="font-semibold">Phone:</span> +94 767878513
                    </p>
                    <p className="text-sm">
                      <span className="font-semibold">Email:</span> chathuranga.herath@globalwavenet.com
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};