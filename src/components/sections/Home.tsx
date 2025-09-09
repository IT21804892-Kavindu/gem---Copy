import React from 'react';
import { Linkedin, Github, Mail, Phone, Download } from 'lucide-react';
import my from '../images/my.jpg';
import cvpdf from '../images/Resume - Kavindu Rajapaksha.pdf';

export const Home: React.FC = () => {
  const handleDownloadClick = () => {
    // Fetch the PDF as a blob and open it in a new tab
    fetch(cvpdf)
      .then(res => res.blob())
      .then(blob => {
        const fileURL = URL.createObjectURL(blob);
        window.open(fileURL, '_blank');
      })
      .catch(error => {
        console.error('Failed to open CV:', error);
      });
  }
  return (
    <section id="home" className="pt-24 pb-16 md:py-32 bg-gradient-to-br from-blue-50 to-white">
      <div className="container mx-auto px-4 md:px-6">
        <div className="flex flex-col md:flex-row items-center">
          <div className="md:w-1/2 mb-8 md:mb-0">
            <div className="relative w-48 h-48 md:w-64 md:h-64 mx-auto md:mx-0 rounded-full overflow-hidden border-4 border-white shadow-lg">
              <img
                // src="https://images.pexels.com/photos/2379004/pexels-photo-2379004.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" 
                src={my}
                alt="Kavindu Rajapaksha"
                className="w-full h-full object-cover"
              />
            </div>
          </div>
          <div className="md:w-1/2 text-center md:text-left">
            <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-2">Kavindu Rajapaksha</h1>
            <h2 className="text-xl md:text-2xl text-blue-600 font-semibold mb-6">Software Engineer</h2>
            <p className="text-gray-600 mb-8 max-w-lg mx-auto md:mx-0">
              {/* A highly motivated and results-oriented software engineer, passionate about developing and implementing innovative solutions for software engineering works. Proven ability to lead and manage teams to deliver projects on time and within budget. Seeking a challenging role where I can leverage my skills and experience to contribute to a dynamic and innovative company. */}
              I am a highly motivated and results-oriented software engineer, passionate about developing and implementing innovative solutions for software engineering works. With experience in both system administration and ERP development, I've built a versatile skill set that allows me to tackle complex challenges.
              <br />
              Currently pursuing my BSc (Hons) in Information Technology at the Sri Lanka Institute of Information & Technology, I've maintained a CGPA of 3.17 and also hold a BSc (Hons) in Civil and Structural Engineering from Liverpool John Moores University.
            </p>
            <div className="flex flex-wrap justify-center md:justify-start gap-4 mb-8">
              <a
                href="https://www.linkedin.com/in/kavindu-rajapaksha-in/"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md transition-colors"
              >
                <Linkedin size={18} /> LinkedIn
              </a>
              <a
                href="mailto:kavindu.kpr@gmail.com"
                className="flex items-center gap-2 bg-teal-600 hover:bg-teal-700 text-white px-4 py-2 rounded-md transition-colors"
              >
                <Mail size={18} /> Contact Me
              </a>
              <a>
                <button
                  onClick={handleDownloadClick}
                  className="flex items-center gap-2 bg-gray-700 hover:bg-gray-800 text-white px-4 py-2 rounded-md transition-colors"
                >
                  <Download size={18} /> Download CV
                </button>
              </a>
            </div>
            <div className="flex justify-center md:justify-start space-x-6">
              <div className="text-center">
                <p className="text-3xl font-bold text-blue-600">1+</p>
                <p className="text-sm text-gray-600">Years Experience</p>
              </div>
              <div className="text-center">
                <p className="text-3xl font-bold text-blue-600">5+</p>
                <p className="text-sm text-gray-600">Projects</p>
              </div>
              <div className="text-center">
                <p className="text-3xl font-bold text-blue-600">10+</p>
                <p className="text-sm text-gray-600">Technologies</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};