import React from 'react';
import { Linkedin, Github, Mail, Phone } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="bg-gray-900 text-white py-12">
      <div className="container mx-auto px-4 md:px-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <h3 className="text-xl font-bold mb-4">Kavindu Rajapaksha</h3>
            <p className="text-gray-300 mb-4 max-w-md">
              A highly motivated and results-oriented software engineer, passionate about developing innovative solutions for software engineering challenges.
            </p>
            <div className="flex space-x-4">
              <a href="https://www.linkedin.com/in/kavindu-rajapaksha-in/" 
                 className="text-gray-300 hover:text-white transition-colors" 
                 target="_blank" rel="noopener noreferrer">
                <Linkedin size={20} />
              </a>
              <a href="mailto:kavindu.kpr@gmail.com" 
                 className="text-gray-300 hover:text-white transition-colors">
                <Mail size={20} />
              </a>
              <a href="tel:+94768565493" 
                 className="text-gray-300 hover:text-white transition-colors">
                <Phone size={20} />
              </a>
              <a href="https://github.com/IT21804892-Kavindu" 
                 className="text-gray-300 hover:text-white transition-colors" 
                 target="_blank" rel="noopener noreferrer">
                <Github size={20} />
              </a>
            </div>
          </div>
          <div>
            <h3 className="text-xl font-bold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              {[
                { name: 'Home', id: 'home' },
                { name: 'Journal', id: 'journal' },
                { name: 'Career Plan', id: 'career-plan' },
                { name: 'Resume', id: 'resume' },
                { name: 'Certifications', id: 'certifications' }
              ].map((item) => (
                <li key={item.id}>
                  <a 
                    href={`#${item.id}`}
                    className="text-gray-300 hover:text-white transition-colors"
                  >
                    {item.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="border-t border-gray-800 mt-8 pt-8 text-sm text-gray-400">
          <p>© {new Date().getFullYear()} Kavindu Rajapaksha. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};