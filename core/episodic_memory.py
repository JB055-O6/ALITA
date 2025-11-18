"""
Long-Term Episodic Memory System

Completes Task 25 requirements:
- Episodic event storage (already in brain.py)
- Preference and personal detail persistence
- Anniversary and milestone tracking
- Skill development tracker
- Encrypted memory export/import

Integrates with:
- Brain module for RAG memory
- LearningSystem for skill tracking

All features are FREE and run locally!
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
from collections import defaultdict

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    import base64
except ImportError:
    Fernet = None
    hashes = None
    PBKDF2 = None
    base64 = None


@dataclass
class PersonalDetail:
    """Personal detail about the user."""
    category: str  # name, birthday, preference, pet_peeve, etc.
    key: str
    value: str
    importance: float
    created: datetime
    last_referenced: datetime
    reference_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['created'] = self.created.isoformat()
        data['last_referenced'] = self.last_referenced.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonalDetail':
        """Create from dictionary."""
        data['created'] = datetime.fromisoformat(data['created'])
        data['last_referenced'] = datetime.fromisoformat(data['last_referenced'])
        return cls(**data)


@dataclass
class Milestone:
    """Milestone or anniversary."""
    milestone_id: str
    title: str
    description: str
    date: datetime
    category: str  # anniversary, achievement, birthday, etc.
    recurring: bool  # True for anniversaries
    acknowledged: bool
    created: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['date'] = self.date.isoformat()
        data['created'] = self.created.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Milestone':
        """Create from dictionary."""
        data['date'] = datetime.fromisoformat(data['date'])
        data['created'] = datetime.fromisoformat(data['created'])
        return cls(**data)


@dataclass
class SkillProgress:
    """Skill development progress."""
    skill_name: str
    category: str  # programming, language, tool, etc.
    level: str  # beginner, intermediate, advanced, expert
    start_date: datetime
    last_practice: datetime
    practice_count: int
    milestones_achieved: List[str]
    notes: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['start_date'] = self.start_date.isoformat()
        data['last_practice'] = self.last_practice.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SkillProgress':
        """Create from dictionary."""
        data['start_date'] = datetime.fromisoformat(data['start_date'])
        data['last_practice'] = datetime.fromisoformat(data['last_practice'])
        return cls(**data)


class PersonalDetailsManager:
    """Manage personal details and preferences."""
    
    def __init__(self, data_file: Path = None):
        self.data_file = data_file or Path("data/personal_details.json")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.details: Dict[str, PersonalDetail] = {}
        self.load_data()
    
    def load_data(self):
        """Load personal details."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    for detail_data in data.get("details", []):
                        detail = PersonalDetail.from_dict(detail_data)
                        self.details[detail.key] = detail
            except Exception as e:
                logging.error(f"Failed to load personal details: {e}")
    
    def save_data(self):
        """Save personal details."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump({
                    "details": [d.to_dict() for d in self.details.values()]
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save personal details: {e}")
    
    def add_detail(self, category: str, key: str, value: str, importance: float = 0.5):
        """Add or update personal detail.
        
        Args:
            category: Detail category
            key: Detail key
            value: Detail value
            importance: Importance (0-1)
        """
        if key in self.details:
            # Update existing
            detail = self.details[key]
            detail.value = value
            detail.last_referenced = datetime.now()
            detail.reference_count += 1
        else:
            # Create new
            detail = PersonalDetail(
                category=category,
                key=key,
                value=value,
                importance=importance,
                created=datetime.now(),
                last_referenced=datetime.now(),
                reference_count=1
            )
            self.details[key] = detail
        
        self.save_data()
        logging.info(f"Added personal detail: {category}/{key}")
    
    def get_detail(self, key: str) -> Optional[str]:
        """Get personal detail value.
        
        Args:
            key: Detail key
            
        Returns:
            Detail value or None
        """
        if key in self.details:
            detail = self.details[key]
            detail.last_referenced = datetime.now()
            detail.reference_count += 1
            self.save_data()
            return detail.value
        return None
    
    def get_by_category(self, category: str) -> List[PersonalDetail]:
        """Get all details in category.
        
        Args:
            category: Category name
            
        Returns:
            List of details
        """
        return [d for d in self.details.values() if d.category == category]
    
    def get_important_details(self, min_importance: float = 0.7) -> List[PersonalDetail]:
        """Get important details.
        
        Args:
            min_importance: Minimum importance threshold
            
        Returns:
            List of important details
        """
        return [d for d in self.details.values() if d.importance >= min_importance]


class MilestoneTracker:
    """Track anniversaries and milestones."""
    
    def __init__(self, data_file: Path = None):
        self.data_file = data_file or Path("data/milestones.json")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.milestones: Dict[str, Milestone] = {}
        self.load_data()
    
    def load_data(self):
        """Load milestones."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    for milestone_data in data.get("milestones", []):
                        milestone = Milestone.from_dict(milestone_data)
                        self.milestones[milestone.milestone_id] = milestone
            except Exception as e:
                logging.error(f"Failed to load milestones: {e}")
    
    def save_data(self):
        """Save milestones."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump({
                    "milestones": [m.to_dict() for m in self.milestones.values()]
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save milestones: {e}")
    
    def add_milestone(self,
                     title: str,
                     description: str,
                     date: datetime,
                     category: str = "achievement",
                     recurring: bool = False) -> str:
        """Add milestone.
        
        Args:
            title: Milestone title
            description: Description
            date: Date of milestone
            category: Category
            recurring: True for anniversaries
            
        Returns:
            Milestone ID
        """
        milestone_id = f"milestone_{datetime.now().timestamp()}"
        
        milestone = Milestone(
            milestone_id=milestone_id,
            title=title,
            description=description,
            date=date,
            category=category,
            recurring=recurring,
            acknowledged=False,
            created=datetime.now()
        )
        
        self.milestones[milestone_id] = milestone
        self.save_data()
        
        logging.info(f"Added milestone: {title}")
        return milestone_id
    
    def get_upcoming_milestones(self, days_ahead: int = 7) -> List[Milestone]:
        """Get upcoming milestones.
        
        Args:
            days_ahead: Days to look ahead
            
        Returns:
            List of upcoming milestones
        """
        now = datetime.now()
        cutoff = now + timedelta(days=days_ahead)
        
        upcoming = []
        
        for milestone in self.milestones.values():
            # Check if milestone is upcoming
            if milestone.recurring:
                # For recurring, check if anniversary is coming up
                next_occurrence = milestone.date.replace(year=now.year)
                if next_occurrence < now:
                    next_occurrence = next_occurrence.replace(year=now.year + 1)
                
                if now <= next_occurrence <= cutoff:
                    upcoming.append(milestone)
            else:
                # For one-time milestones
                if now <= milestone.date <= cutoff and not milestone.acknowledged:
                    upcoming.append(milestone)
        
        return sorted(upcoming, key=lambda m: m.date)
    
    def acknowledge_milestone(self, milestone_id: str):
        """Mark milestone as acknowledged.
        
        Args:
            milestone_id: Milestone ID
        """
        if milestone_id in self.milestones:
            self.milestones[milestone_id].acknowledged = True
            self.save_data()
            logging.info(f"Acknowledged milestone: {milestone_id}")
    
    def get_anniversaries(self) -> List[Milestone]:
        """Get all anniversaries.
        
        Returns:
            List of recurring milestones
        """
        return [m for m in self.milestones.values() if m.recurring]


class SkillDevelopmentTracker:
    """Track skill development and progress."""
    
    def __init__(self, data_file: Path = None):
        self.data_file = data_file or Path("data/skill_development.json")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.skills: Dict[str, SkillProgress] = {}
        self.load_data()
    
    def load_data(self):
        """Load skill progress."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    for skill_data in data.get("skills", []):
                        skill = SkillProgress.from_dict(skill_data)
                        self.skills[skill.skill_name] = skill
            except Exception as e:
                logging.error(f"Failed to load skill development: {e}")
    
    def save_data(self):
        """Save skill progress."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump({
                    "skills": [s.to_dict() for s in self.skills.values()]
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save skill development: {e}")
    
    def start_learning_skill(self,
                            skill_name: str,
                            category: str = "general",
                            level: str = "beginner"):
        """Start learning a new skill.
        
        Args:
            skill_name: Skill name
            category: Skill category
            level: Starting level
        """
        if skill_name not in self.skills:
            skill = SkillProgress(
                skill_name=skill_name,
                category=category,
                level=level,
                start_date=datetime.now(),
                last_practice=datetime.now(),
                practice_count=0,
                milestones_achieved=[],
                notes=""
            )
            self.skills[skill_name] = skill
            self.save_data()
            logging.info(f"Started learning: {skill_name}")
    
    def record_practice(self, skill_name: str, notes: str = ""):
        """Record skill practice.
        
        Args:
            skill_name: Skill name
            notes: Practice notes
        """
        if skill_name in self.skills:
            skill = self.skills[skill_name]
            skill.last_practice = datetime.now()
            skill.practice_count += 1
            if notes:
                skill.notes = notes
            
            # Check for level progression
            self._check_level_progression(skill)
            
            self.save_data()
    
    def _check_level_progression(self, skill: SkillProgress):
        """Check if skill level should progress."""
        # Simple progression based on practice count
        if skill.practice_count >= 100 and skill.level == "beginner":
            skill.level = "intermediate"
            skill.milestones_achieved.append(f"Reached intermediate level on {datetime.now().date()}")
            logging.info(f"Skill progression: {skill.skill_name} → intermediate")
        
        elif skill.practice_count >= 500 and skill.level == "intermediate":
            skill.level = "advanced"
            skill.milestones_achieved.append(f"Reached advanced level on {datetime.now().date()}")
            logging.info(f"Skill progression: {skill.skill_name} → advanced")
        
        elif skill.practice_count >= 1000 and skill.level == "advanced":
            skill.level = "expert"
            skill.milestones_achieved.append(f"Reached expert level on {datetime.now().date()}")
            logging.info(f"Skill progression: {skill.skill_name} → expert")
    
    def get_skill_progress(self, skill_name: str) -> Optional[SkillProgress]:
        """Get skill progress.
        
        Args:
            skill_name: Skill name
            
        Returns:
            Skill progress or None
        """
        return self.skills.get(skill_name)
    
    def get_recent_progress(self, days: int = 30) -> List[SkillProgress]:
        """Get recently practiced skills.
        
        Args:
            days: Days to look back
            
        Returns:
            List of recently practiced skills
        """
        cutoff = datetime.now() - timedelta(days=days)
        return [s for s in self.skills.values() if s.last_practice >= cutoff]
    
    def celebrate_progress(self, skill_name: str) -> Optional[str]:
        """Generate celebration message for progress.
        
        Args:
            skill_name: Skill name
            
        Returns:
            Celebration message or None
        """
        skill = self.skills.get(skill_name)
        if not skill:
            return None
        
        days_learning = (datetime.now() - skill.start_date).days
        
        messages = []
        
        if skill.practice_count % 100 == 0:
            messages.append(f"🎉 {skill.practice_count} practice sessions with {skill.skill_name}!")
        
        if days_learning % 30 == 0:
            months = days_learning // 30
            messages.append(f"📅 {months} months of learning {skill.skill_name}!")
        
        if skill.milestones_achieved:
            latest = skill.milestones_achieved[-1]
            messages.append(f"🏆 Latest achievement: {latest}")
        
        return " ".join(messages) if messages else None


class MemoryExporter:
    """Export and import memory with encryption."""
    
    def __init__(self):
        self.encryption_available = Fernet is not None
    
    def generate_key(self, password: str) -> bytes:
        """Generate encryption key from password.
        
        Args:
            password: User password
            
        Returns:
            Encryption key
        """
        if not self.encryption_available:
            raise RuntimeError("Encryption not available")
        
        # Use PBKDF2 to derive key from password
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'alita_memory_salt',  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def export_memory(self,
                     personal_details: PersonalDetailsManager,
                     milestones: MilestoneTracker,
                     skills: SkillDevelopmentTracker,
                     output_file: Path,
                     password: Optional[str] = None) -> bool:
        """Export memory to file.
        
        Args:
            personal_details: Personal details manager
            milestones: Milestone tracker
            skills: Skill development tracker
            output_file: Output file path
            password: Optional encryption password
            
        Returns:
            True if successful
        """
        try:
            # Collect all data
            data = {
                "export_date": datetime.now().isoformat(),
                "personal_details": [d.to_dict() for d in personal_details.details.values()],
                "milestones": [m.to_dict() for m in milestones.milestones.values()],
                "skills": [s.to_dict() for s in skills.skills.values()]
            }
            
            # Convert to JSON
            json_data = json.dumps(data, indent=2)
            
            # Encrypt if password provided
            if password and self.encryption_available:
                key = self.generate_key(password)
                fernet = Fernet(key)
                encrypted_data = fernet.encrypt(json_data.encode())
                
                with open(output_file, 'wb') as f:
                    f.write(encrypted_data)
                
                logging.info(f"Exported encrypted memory to {output_file}")
            else:
                with open(output_file, 'w') as f:
                    f.write(json_data)
                
                logging.info(f"Exported memory to {output_file}")
            
            return True
            
        except Exception as e:
            logging.error(f"Memory export failed: {e}")
            return False
    
    def import_memory(self,
                     input_file: Path,
                     password: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Import memory from file.
        
        Args:
            input_file: Input file path
            password: Optional decryption password
            
        Returns:
            Imported data or None
        """
        try:
            # Read file
            if password and self.encryption_available:
                with open(input_file, 'rb') as f:
                    encrypted_data = f.read()
                
                # Decrypt
                key = self.generate_key(password)
                fernet = Fernet(key)
                json_data = fernet.decrypt(encrypted_data).decode()
            else:
                with open(input_file, 'r') as f:
                    json_data = f.read()
            
            # Parse JSON
            data = json.loads(json_data)
            
            logging.info(f"Imported memory from {input_file}")
            return data
            
        except Exception as e:
            logging.error(f"Memory import failed: {e}")
            return None


class EpisodicMemorySystem:
    """Main long-term episodic memory system."""
    
    def __init__(self):
        self.personal_details = PersonalDetailsManager()
        self.milestones = MilestoneTracker()
        self.skills = SkillDevelopmentTracker()
        self.exporter = MemoryExporter()
        
        logging.info("Episodic Memory System initialized")
    
    def add_personal_detail(self, category: str, key: str, value: str, importance: float = 0.5):
        """Add personal detail."""
        self.personal_details.add_detail(category, key, value, importance)
    
    def get_personal_detail(self, key: str) -> Optional[str]:
        """Get personal detail."""
        return self.personal_details.get_detail(key)
    
    def add_milestone(self, title: str, description: str, date: datetime,
                     category: str = "achievement", recurring: bool = False) -> str:
        """Add milestone."""
        return self.milestones.add_milestone(title, description, date, category, recurring)
    
    def check_upcoming_milestones(self) -> List[Milestone]:
        """Check for upcoming milestones."""
        return self.milestones.get_upcoming_milestones(days_ahead=7)
    
    def start_learning(self, skill_name: str, category: str = "general"):
        """Start learning a skill."""
        self.skills.start_learning_skill(skill_name, category)
    
    def record_skill_practice(self, skill_name: str, notes: str = ""):
        """Record skill practice."""
        self.skills.record_practice(skill_name, notes)
        
        # Check for celebration
        celebration = self.skills.celebrate_progress(skill_name)
        if celebration:
            logging.info(f"Celebration: {celebration}")
            return celebration
        return None
    
    def export_memories(self, output_file: Path, password: Optional[str] = None) -> bool:
        """Export all memories."""
        return self.exporter.export_memory(
            self.personal_details,
            self.milestones,
            self.skills,
            output_file,
            password
        )
    
    def import_memories(self, input_file: Path, password: Optional[str] = None) -> bool:
        """Import memories."""
        data = self.exporter.import_memory(input_file, password)
        
        if not data:
            return False
        
        try:
            # Import personal details
            for detail_data in data.get("personal_details", []):
                detail = PersonalDetail.from_dict(detail_data)
                self.personal_details.details[detail.key] = detail
            self.personal_details.save_data()
            
            # Import milestones
            for milestone_data in data.get("milestones", []):
                milestone = Milestone.from_dict(milestone_data)
                self.milestones.milestones[milestone.milestone_id] = milestone
            self.milestones.save_data()
            
            # Import skills
            for skill_data in data.get("skills", []):
                skill = SkillProgress.from_dict(skill_data)
                self.skills.skills[skill.skill_name] = skill
            self.skills.save_data()
            
            logging.info("Memory import completed")
            return True
            
        except Exception as e:
            logging.error(f"Memory import processing failed: {e}")
            return False
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory summary."""
        return {
            "personal_details_count": len(self.personal_details.details),
            "milestones_count": len(self.milestones.milestones),
            "skills_tracked": len(self.skills.skills),
            "upcoming_milestones": len(self.check_upcoming_milestones()),
            "recent_skill_progress": len(self.skills.get_recent_progress(30))
        }


# Global instance
_episodic_memory: Optional[EpisodicMemorySystem] = None


def get_episodic_memory() -> EpisodicMemorySystem:
    """Get global episodic memory instance."""
    global _episodic_memory
    
    if _episodic_memory is None:
        _episodic_memory = EpisodicMemorySystem()
    
    return _episodic_memory
