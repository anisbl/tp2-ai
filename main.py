import streamlit as st
import time
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import random
import copy
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class MultiGroupTimeTableCSP:
    def __init__(self, num_groups=6):
        # Days and time slots
        self.days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
        self.slots_per_day = {"Sunday": 5, "Monday": 5, "Tuesday": 3, "Wednesday": 5, "Thursday": 5}
        self.num_groups = num_groups
        
        # Generate all possible time slots
        self.all_slots = []
        for day in self.days:
            for slot in range(1, self.slots_per_day[day] + 1):
                self.all_slots.append((day, slot))
        
        # Courses data - will be populated later
        self.courses = []
        
        # Variables will be (course_name, component, group_number)
        # For lectures, group_number will be 0 (shared across all groups)
        self.variables = []
        self.domains = {}
        
    def initialize_variables_and_domains(self):
        """Initialize variables and domains after courses are set"""
        self.variables = []
        for course in self.courses:
            for component in course["components"]:
                if component == "lecture":
                    # Lectures are shared across all groups
                    self.variables.append((course["name"], component, 0))
                else:
                    # TD and TP are per group
                    for group in range(1, self.num_groups + 1):
                        self.variables.append((course["name"], component, group))
        
        self.domains = {var: list(self.all_slots) for var in self.variables}
    
    def get_teacher_for_component(self, course_name, component, group):
        course = next(c for c in self.courses if c["name"] == course_name)
        
        # For lectures, use the main teacher
        if component == "lecture":
            return course["teacher"]
        
        # For TP, check if specific TP teachers are defined
        if component == "tp" and "tp_teachers" in course:
            # If tp_teachers is a dict with group keys, use that
            if isinstance(course["tp_teachers"], dict):
                return course["tp_teachers"].get(group, course["tp_teachers"].get(1, course["teacher"]))
            # If tp_teachers is a list, use the first one or group-specific one if available
            elif isinstance(course["tp_teachers"], list):
                if len(course["tp_teachers"]) >= group:
                    return course["tp_teachers"][group-1]
                return course["tp_teachers"][0]
            else:
                return course["tp_teachers"]
        
        # For TD, check if specific TD teachers are defined
        if component == "td" and "td_teachers" in course:
            if isinstance(course["td_teachers"], dict):
                return course["td_teachers"].get(group, course["td_teachers"].get(1, course["teacher"]))
            elif isinstance(course["td_teachers"], list):
                if len(course["td_teachers"]) >= group:
                    return course["td_teachers"][group-1]
                return course["td_teachers"][0]
            else:
                return course["td_teachers"]
        
        # Default to main teacher
        return course["teacher"]
    
    def check_teacher_workdays(self, assignment):
        teacher_days = defaultdict(set)
        for (course, component, group), slot in assignment.items():
            teacher = self.get_teacher_for_component(course, component, group)
            teacher_days[teacher].add(slot[0])  # Add the day
        
        # Check if any teacher works more than 2 days
        return all(len(days) <= 2 for days in teacher_days.values())
    
    def max_three_successive_slots(self, assignment):
        # Track slots per day per group
        group_day_slots = defaultdict(lambda: defaultdict(list))
        
        for (course, component, group), slot in assignment.items():
            day, time_slot = slot
            
            # For lectures (group 0), add the slot to all groups' schedules
            if group == 0:
                for g in range(1, self.num_groups + 1):
                    group_day_slots[g][day].append(time_slot)
            else:
                group_day_slots[group][day].append(time_slot)
        
        # Check each group's day schedule
        for group, day_dict in group_day_slots.items():
            for day, slots in day_dict.items():
                slots.sort()
                for i in range(len(slots) - 3):
                    # Check if there are 4 consecutive slots
                    if slots[i+3] - slots[i] == 3:
                        return False
        return True
    
    def is_consistent(self, var, value, assignment):
        """Check if assigning value to var is consistent with current assignment"""
        course, component, group = var
        day, slot = value
        
        for assigned_var, assigned_val in assignment.items():
            assigned_course, assigned_component, assigned_group = assigned_var
            assigned_day, assigned_slot = assigned_val
            
            # Same time slot conflicts
            if assigned_val == value:
                # Same group can't have two different sessions at the same time
                # (including lectures which affect all groups)
                if assigned_group == group or assigned_group == 0 or group == 0:
                    return False
                
                # Same course can't have multiple components at the same time
                if assigned_course == course:
                    return False
            
            # Teacher conflicts - same teacher can't teach two different sessions at the same time
            if assigned_day == day and assigned_slot == slot:
                teacher1 = self.get_teacher_for_component(course, component, group)
                teacher2 = self.get_teacher_for_component(assigned_course, assigned_component, assigned_group)
                if teacher1 == teacher2:
                    return False
        
        # Check max three successive slots constraint
        temp_assignment = assignment.copy()
        temp_assignment[var] = value
        if not self.max_three_successive_slots(temp_assignment):
            return False
            
        return True
    
    def ac3(self):
        queue = [(xi, xj) for xi in self.variables for xj in self.variables if xi != xj]
        while queue:
            xi, xj = queue.pop(0)
            if self.revise(xi, xj):
                if len(self.domains[xi]) == 0:
                    return False
                for xk in self.variables:
                    if xk != xi and xk != xj:
                        queue.append((xk, xi))
        return True
    
    def revise(self, xi, xj):
        revised = False
        to_remove = []
        
        xi_course, xi_component, xi_group = xi
        xj_course, xj_component, xj_group = xj
        
        for x in self.domains[xi][:]:
            all_conflict = True
            
            for y in self.domains[xj]:
                # Skip if same variable
                if xi == xj:
                    continue
                
                # Check constraints
                if x == y:
                    # Can't schedule at same time if:
                    # 1. Same group
                    # 2. One is a lecture (affects all groups)
                    # 3. Same course different components
                    if xi_group == xj_group or xi_group == 0 or xj_group == 0 or xi_course == xj_course:
                        continue
                
                if x[0] == y[0] and x[1] == y[1]:  # Same day and time
                    teacher1 = self.get_teacher_for_component(xi_course, xi_component, xi_group)
                    teacher2 = self.get_teacher_for_component(xj_course, xj_component, xj_group)
                    if teacher1 == teacher2:
                        # Same teacher can't teach two sessions at the same time
                        continue
                
                # No conflict found for this pair
                all_conflict = False
                break
            
            if all_conflict:
                to_remove.append(x)
                revised = True
        
        for x in to_remove:
            if x in self.domains[xi]:
                self.domains[xi].remove(x)
        
        return revised
    
    def select_unassigned_variable_mrv(self, assignment):
        unassigned = [var for var in self.variables if var not in assignment]
        
        # Prioritize lectures (they affect all groups)
        lectures = [var for var in unassigned if var[1] == "lecture"]
        if lectures:
            return min(lectures, key=lambda var: len(self.domains[var]))
        
        return min(unassigned, key=lambda var: len(self.domains[var]))
    
    def order_domain_values_lcv(self, var, assignment):
        def count_conflicts(value):
            conflicts = 0
            for other_var in self.variables:
                if other_var != var and other_var not in assignment:
                    conflicts += sum(1 for other_val in self.domains[other_var] if value == other_val)
            return conflicts
        
        return sorted(self.domains[var], key=count_conflicts)
    
    def backtracking_search(self):
        return self.backtrack({})
    
    def backtrack(self, assignment):
        if len(assignment) == len(self.variables):
            return assignment
        
        var = self.select_unassigned_variable_mrv(assignment)
        for value in self.order_domain_values_lcv(var, assignment):
            if self.is_consistent(var, value, assignment):
                assignment[var] = value
                
                # Forward checking
                pruned = self.forward_check(var, value, assignment)
                if pruned is None:  # Domain wipeout occurred
                    continue
                
                result = self.backtrack(assignment)
                if result is not None:
                    return result
                
                # Restore pruned values
                self.restore_domains(pruned)
                del assignment[var]
        
        return None
    
    def forward_check(self, var, value, assignment):
        """Prune domain values that would conflict with the new assignment"""
        pruned = defaultdict(list)
        course, component, group = var
        day, slot = value
        
        for other_var in self.variables:
            if other_var not in assignment and other_var != var:
                other_course, other_component, other_group = other_var
                
                for other_val in list(self.domains[other_var]):
                    other_day, other_slot = other_val
                    
                    # Check for time conflicts
                    if other_val == value:
                        # Same time conflicts if:
                        # 1. Same group
                        # 2. One is a lecture (affects all groups)
                        # 3. Same course different components
                        if other_group == group or other_group == 0 or group == 0 or other_course == course:
                            pruned[other_var].append(other_val)
                            self.domains[other_var].remove(other_val)
                            continue
                    
                    # Check for teacher conflicts
                    if other_day == day and other_slot == slot:
                        teacher1 = self.get_teacher_for_component(course, component, group)
                        teacher2 = self.get_teacher_for_component(other_course, other_component, other_group)
                        if teacher1 == teacher2:
                            pruned[other_var].append(other_val)
                            self.domains[other_var].remove(other_val)
                
                # Check if domain became empty
                if not self.domains[other_var]:
                    # Restore all pruned values before returning
                    self.restore_domains(pruned)
                    return None
        
        return pruned
    
    def restore_domains(self, pruned):
        """Restore pruned domain values"""
        for var, values in pruned.items():
            for value in values:
                if value not in self.domains[var]:
                    self.domains[var].append(value)
    
    def solve(self, timeout=30):
        start_time = time.time()
        
        # Initialize variables and domains
        self.initialize_variables_and_domains()
        
        if self.ac3():
            solution = self.backtracking_search_with_timeout(start_time, timeout)
        else:
            solution = None
            
        end_time = time.time()
        
        if solution:
            # Try to optimize for soft constraint (teacher max 2 workdays)
            optimized_solution = self.optimize_teacher_workdays(solution)
            teacher_days = self.count_teacher_workdays(optimized_solution)
            
            return {
                "solution": optimized_solution,
                "time": end_time - start_time,
                "teacher_days": teacher_days,
                "soft_constraint_satisfied": self.check_teacher_workdays(optimized_solution)
            }
        else:
            return {
                "solution": None,
                "time": end_time - start_time
            }
    
    def backtracking_search_with_timeout(self, start_time, timeout):
        """Version of backtracking search that times out after specified seconds"""
        def backtrack_timed(assignment):
            if time.time() - start_time > timeout:
                return "TIMEOUT"
                
            if len(assignment) == len(self.variables):
                return assignment
            
            var = self.select_unassigned_variable_mrv(assignment)
            for value in self.order_domain_values_lcv(var, assignment):
                if self.is_consistent(var, value, assignment):
                    assignment[var] = value
                    
                    # Forward checking
                    pruned = self.forward_check(var, value, assignment)
                    if pruned is None:  # Domain wipeout occurred
                        continue
                    
                    result = backtrack_timed(assignment)
                    if result == "TIMEOUT":
                        return "TIMEOUT"
                    if result is not None:
                        return result
                    
                    # Restore pruned values
                    self.restore_domains(pruned)
                    del assignment[var]
            
            return None
        
        result = backtrack_timed({})
        if result == "TIMEOUT":
            st.warning(f"Search timed out after {timeout} seconds. Returning best partial solution.")
            return None
        return result
    
    def count_teacher_workdays(self, assignment):
        teacher_days = defaultdict(set)
        for (course, component, group), slot in assignment.items():
            teacher = self.get_teacher_for_component(course, component, group)
            teacher_days[teacher].add(slot[0])
        
        return {teacher: len(days) for teacher, days in teacher_days.items()}
    
    def evaluate_solution(self, solution):
        """Lower score is better"""
        teacher_days = self.count_teacher_workdays(solution)
        
        # Calculate penalties
        teacher_days_penalty = sum(max(0, days - 2) * 10 for days in teacher_days.values())
        
        return teacher_days_penalty
    
    def optimize_teacher_workdays(self, solution):
        # Local search to optimize teacher workdays
        best_solution = copy.deepcopy(solution)
        best_score = self.evaluate_solution(solution)
        
        temperature = 1.0
        cooling_rate = 0.97
        iterations = 1000
        
        current_solution = copy.deepcopy(solution)
        current_score = best_score
        
        for i in range(iterations):
            # Choose random variables to swap that are not lectures
            non_lectures = [(var, val) for var, val in current_solution.items() if var[1] != "lecture"]
            if len(non_lectures) < 2:
                break
                
            var1, val1 = random.choice(non_lectures)
            var2, val2 = random.choice(non_lectures)
            course1, component1, group1 = var1
            course2, component2, group2 = var2
            
            # Only swap if they're the same component type and same group
            # This preserves the constraint that lectures are shared
            if component1 == component2 and group1 == group2:
                neighbor = copy.deepcopy(current_solution)
                neighbor[var1], neighbor[var2] = neighbor[var2], neighbor[var1]
                
                # Check if neighbor satisfies hard constraints
                if self.max_three_successive_slots(neighbor):
                    neighbor_score = self.evaluate_solution(neighbor)
                    
                    # Calculate acceptance probability
                    delta = current_score - neighbor_score
                    if delta > 0 or random.random() < math.exp(delta / temperature):
                        current_solution = neighbor
                        current_score = neighbor_score
                        
                        if current_score < best_score:
                            best_solution = copy.deepcopy(current_solution)
                            best_score = current_score
            
            # Cool down temperature
            temperature *= cooling_rate
        
        return best_solution


def create_timetable_df(solution, csp, group):
    # Create empty dataframe
    days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
    max_slots = 5
    
    # Initialize with empty strings
    df = pd.DataFrame(index=range(1, max_slots+1), columns=days)
    df.fillna("", inplace=True)
    
    # Fill the dataframe with sessions for the specified group and shared lectures
    for (course, component, assigned_group), (day, slot) in solution.items():
        # Include lectures (group 0) and specific group sessions
        if assigned_group == 0 or assigned_group == group:
            # Get teacher info
            teacher = csp.get_teacher_for_component(course, component, assigned_group)
            if isinstance(teacher, list):
                teacher = teacher[0]  # Simplify for display
                
            component_label = "Lecture" if component == "lecture" else component.upper()
            current = df.at[slot, day]
            if current:
                df.at[slot, day] = f"{current}\n{course} ({component_label})\n{teacher}"
            else:
                df.at[slot, day] = f"{course} ({component_label})\n{teacher}"
    
    return df


def visualize_timetable(solution, csp, group):
    # Create a timetable DataFrame for the specified group
    df = create_timetable_df(solution, csp, group)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Hide the axes
    ax.axis('tight')
    ax.axis('off')
    
    # Create color mapping for courses
    courses = set()
    for (course, component, assigned_group) in solution.keys():
        if assigned_group == 0 or assigned_group == group:
            courses.add(course)
    
    colors = list(mcolors.TABLEAU_COLORS)
    color_map = {course: colors[i % len(colors)] for i, course in enumerate(sorted(courses))}
    
    # Create cell colors and track component types for cell styling
    cell_colors = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        for idx in df.index:
            cell_val = df.at[idx, col]
            if cell_val:
                course = cell_val.split(' (')[0]
                cell_colors.at[idx, col] = color_map.get(course, 'white')
            else:
                cell_colors.at[idx, col] = 'white'

    # Convert to numpy array for matplotlib
    cell_colors = cell_colors.to_numpy()
    
    # Create table
    table = ax.table(cellText=df.to_numpy(), 
                     rowLabels=df.index, 
                     colLabels=df.columns,
                     cellColours=cell_colors,
                     cellLoc='center', 
                     loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Add group label
    plt.title(f"Timetable for Group {group}", size=16)
    
    return fig


def main():
    st.set_page_config(layout="wide", page_title="Multi-Group Timetable CSP Solver")
    
    st.title("Constraint Satisfaction Problem for Multi-Group Timetable Scheduling")
    st.markdown("Lectures are shared across all groups, while TD (tutorials) and TP (labs) are scheduled separately for each group.")
    
    # Number of groups configuration
    st.sidebar.header("Configuration")
    num_groups = st.sidebar.number_input("Number of Groups", min_value=1, max_value=10, value=6)
    
    # Course configuration
    st.sidebar.header("Course Configuration")
    
    # Default courses
    default_courses = [
        {"name": "Sécurité", "teacher": "Teacher 1"},
        {"name": "Méthodes formelles", "teacher": "Teacher 2"},
        {"name": "Analyse numérique", "teacher": "Teacher 3"},
        {"name": "Programmation avancée", "teacher": "Teacher 4"},
        {"name": "BD répartie", "teacher": "Teacher 5"},
    ]
    
    # Allow adding more courses
    num_courses = st.sidebar.number_input("Number of courses", min_value=3, max_value=10, value=5)
    
    courses = []
    for i in range(num_courses):
        st.sidebar.subheader(f"Course {i+1}")
        
        if i < len(default_courses):
            name = st.sidebar.text_input(f"Course {i+1} Name", value=default_courses[i]["name"], key=f"course_name_{i}")
            teacher = st.sidebar.text_input(f"Course {i+1} Teacher", value=default_courses[i]["teacher"], key=f"course_teacher_{i}")
        else:
            name = st.sidebar.text_input(f"Course {i+1} Name", value=f"Course {i+1}", key=f"course_name_{i}")
            teacher = st.sidebar.text_input(f"Course {i+1} Teacher", value=f"Teacher {i+1}", key=f"course_teacher_{i}")
        
        # Component selection
        components = []
        has_lecture = st.sidebar.checkbox(f"Has Lecture", value=True, key=f"has_lecture_{i}")
        if has_lecture:
            components.append("lecture")
        
        has_td = st.sidebar.checkbox(f"Has TD (Tutorials)", value=True, key=f"has_td_{i}")
        if has_td:
            components.append("td")
            td_teachers = st.sidebar.text_input(
                f"TD Teachers (same for all groups or comma-separated list for each group)", 
                value=teacher, 
                key=f"td_teachers_{i}"
            )
            
        has_tp = st.sidebar.checkbox(f"Has TP (Lab)", value=False, key=f"has_tp_{i}")
        if has_tp:
            components.append("tp")
            tp_teachers = st.sidebar.text_input(
                f"TP Teachers (same for all groups or comma-separated list for each group)", 
                value=teacher, 
                key=f"tp_teachers_{i}"
            )
        
        course_data = {
            "name": name,
            "components": components,
            "teacher": teacher
        }
        
        if has_td and td_teachers != teacher:
            if "," in td_teachers:
                course_data["td_teachers"] = [t.strip() for t in td_teachers.split(",")]
            else:
                course_data["td_teachers"] = td_teachers
            
        if has_tp:
            if "," in tp_teachers:
                course_data["tp_teachers"] = [t.strip() for t in tp_teachers.split(",")]
            else:
                course_data["tp_teachers"] = tp_teachers
            
        courses.append(course_data)
    
    # Solver settings
    st.sidebar.header("Solver Settings")
    timeout = st.sidebar.slider("Timeout (seconds)", min_value=10, max_value=300, value=60)
    
    if st.button("Generate Timetables"):
        with st.spinner("Solving CSP (this may take a while)..."):
            csp = MultiGroupTimeTableCSP(num_groups=num_groups)
            csp.courses = courses
            
            # Solve with timeout
            result = csp.solve(timeout=timeout)
            
            if result["solution"]:
                st.success(f"Solution found in {result['time']:.4f} seconds!")
                
                # Display lecture schedule first
                st.subheader("Lecture Schedule (Shared by All Groups)")
                lecture_schedule = {var: val for var, val in result["solution"].items() if var[1] == "lecture"}
                
                if lecture_schedule:
                    # Create empty lecture dataframe
                    days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
                    max_slots = 5
                    lecture_df = pd.DataFrame(index=range(1, max_slots+1), columns=days)
                    lecture_df.fillna("", inplace=True)
                    
                    # Fill with lecture data
                    for (course, component, _), (day, slot) in lecture_schedule.items():
                        teacher = csp.get_teacher_for_component(course, component, 0)
                        lecture_df.at[slot, day] = f"{course} (Lecture)\n{teacher}"
                    
                    st.dataframe(lecture_df, height=250)
                else:
                    st.info("No lectures scheduled.")
                
                # Create tabs for each group
                tabs = st.tabs([f"Group {i+1}" for i in range(num_groups)])
                
                for i, tab in enumerate(tabs):
                    group_num = i + 1
                    with tab:
                        st.subheader(f"Timetable for Group {group_num}")
                        timetable_df = create_timetable_df(result["solution"], csp, group_num)
                        st.dataframe(timetable_df, height=300)
                        
                        st.subheader("Visual Timetable")
                        fig = visualize_timetable(result["solution"], csp, group_num)
                        st.pyplot(fig)
                
                # Display teacher workday information
                st.subheader("Teacher Workdays")
                teacher_days = result["teacher_days"]
                teacher_df = pd.DataFrame(list(teacher_days.items()), columns=["Teacher", "Number of Workdays"])
                st.dataframe(teacher_df)
                
                # Check soft constraint
                if result["soft_constraint_satisfied"]:
                    st.success("Soft Constraint Satisfied: All teachers work 2 days or fewer")
                else:
                    st.warning("Soft Constraint Not Fully Satisfied: Some teachers work more than 2 days")
                    
                # Show statistics
                st.subheader("Statistics")
                stats = {
                    "Execution Time": f"{result['time']:.4f} seconds",
                    "Number of Sessions": len(result["solution"]),
                    "Lectures (shared)": len([v for v in result["solution"].keys() if v[1] == "lecture"]),
                    "TD Sessions": len([v for v in result["solution"].keys() if v[1] == "td"]),
                    "TP Sessions": len([v for v in result["solution"].keys() if v[1] == "tp"]),
                    "Average Teacher Workdays": f"{sum(teacher_days.values()) / len(teacher_days):.2f} days",
                    "Maximum Teacher Workdays": f"{max(teacher_days.values())} days"
                }
                st.json(stats)
                
            else:
                st.error("No solution found! Try relaxing some constraints or increasing the timeout.")
                st.write(f"Time elapsed: {result['time']:.4f} seconds")
                st.info("Tips: Reduce the number of courses or groups, increase the timeout, or simplify the constraints.")

if __name__ == "__main__":
    main()