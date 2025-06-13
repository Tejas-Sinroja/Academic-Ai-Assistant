# Dashboard Feature - Technical Documentation

## Implementation Overview

The Dashboard feature in the Academic AI Assistant is implemented as a Streamlit page function `dashboard_page()` that provides comprehensive analytics and administrative capabilities. This document outlines the technical implementation details.

## Function Structure

The `dashboard_page()` function is organized into several sections:

1. **Header and Introduction**: Sets up the page title and description
2. **Tab Creation**: Establishes four main tabs using Streamlit's tab interface
3. **Database Connection**: Initializes connection to PostgreSQL database
4. **Summary Metrics Retrieval**: Queries basic count metrics from database tables
5. **Tab-specific Content Rendering**: Populates each tab with relevant visualizations
6. **User Management Section**: Provides administrative functions

## Database Queries

### System Overview Metrics

```python
# Get total number of students
cursor.execute("SELECT COUNT(*) FROM students")
total_students = cursor.fetchone()[0]

# Get total number of notes
cursor.execute("SELECT COUNT(*) FROM notes")
total_notes = cursor.fetchone()[0]

# Get total number of tasks
cursor.execute("SELECT COUNT(*) FROM tasks")
total_tasks = cursor.fetchone()[0]

# Get total number of quizzes
cursor.execute("SELECT COUNT(*) FROM quizzes")
total_quizzes = cursor.fetchone()[0]
```

### Activity Tracking Queries

```python
# Get activity by date (last 30 days)
cursor.execute("""
    SELECT 
        DATE(created_at) as date,
        COUNT(*) as count
    FROM notes
    WHERE created_at >= NOW() - INTERVAL '30 days'
    GROUP BY DATE(created_at)
    ORDER BY DATE(created_at)
""")
```

### Student Leaderboard Query

```python
cursor.execute("""
    SELECT 
        s.id,
        s.name,
        s.learning_style,
        COUNT(DISTINCT n.id) as notes_count,
        COUNT(DISTINCT t.id) as tasks_count,
        COUNT(DISTINCT CASE WHEN t.status = 'completed' THEN t.id END) as completed_tasks
    FROM 
        students s
        LEFT JOIN notes n ON s.id = n.student_id
        LEFT JOIN tasks t ON s.id = t.student_id
    GROUP BY 
        s.id, s.name, s.learning_style
    ORDER BY 
        notes_count DESC, completed_tasks DESC
""")
```

### Quiz Performance Query

```python
cursor.execute("""
    SELECT 
        student_id,
        AVG(score_percentage) as avg_score,
        COUNT(*) as quiz_count
    FROM quizzes
    GROUP BY student_id
""")
```

## Data Processing and Visualization

### Activity Chart Generation

```python
# Create activity dataframe
if notes_activity or tasks_activity:
    activity_dates = set()
    notes_dict = {}
    tasks_dict = {}
    
    for date, count in notes_activity:
        activity_dates.add(date)
        notes_dict[date] = count
    
    for date, count in tasks_activity:
        activity_dates.add(date)
        tasks_dict[date] = count
    
    activity_data = []
    for date in sorted(activity_dates):
        activity_data.append({
            "Date": date,
            "Notes Created": notes_dict.get(date, 0),
            "Tasks Created": tasks_dict.get(date, 0)
        })
    
    if activity_data:
        activity_df = pd.DataFrame(activity_data)
        
        # Plot activity chart
        st.line_chart(
            activity_df.set_index("Date")[["Notes Created", "Tasks Created"]]
        )
```

### Leaderboard Score Calculation

```python
# Calculate productivity score
productivity = notes_count * 5 + completed_tasks * 3

# Get quiz data if available
avg_quiz_score = 0
quiz_count = 0
if student_id in quiz_data:
    avg_quiz_score = float(quiz_data[student_id]["avg_score"])
    quiz_count = quiz_data[student_id]["quiz_count"]

# Calculate overall score
overall_score = productivity
if quiz_count > 0:
    overall_score += avg_quiz_score * 0.2
```

### Quiz Performance Trend Analysis

```python
# Convert Score from Decimal to float for calculations
quiz_df["Score"] = quiz_df["Score"].astype(float)

# Create scatter plot with trend line
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(range(len(quiz_df)), quiz_df["Score"], label="Quiz Scores")

# Add trend line if there are enough points
if len(quiz_df) >= 2:
    z = np.polyfit(range(len(quiz_df)), quiz_df["Score"], 1)
    p = np.poly1d(z)
    ax.plot(range(len(quiz_df)), p(range(len(quiz_df))), "r--", label="Trend")
    
    # Determine if improving or declining
    trend_direction = "improving" if z[0] > 0 else "declining"
    st.info(f"Overall quiz performance is **{trend_direction}**. Trend slope: {z[0]:.2f}% per quiz.")
```

## Student Deletion Process

The student deletion process follows these steps:

1. **Selection**: User selects a student from dropdown menu
2. **Confirmation**: User must check a confirmation checkbox
3. **Data Summary**: System displays counts of data to be deleted
4. **Deletion Process**:
   - Creates a fresh database connection to avoid transaction conflicts
   - Deletes related data in sequence: quizzes, tasks, notes, knowledge_base entries
   - Finally deletes the student record
   - Commits after each step to avoid transaction timeouts
5. **Feedback**: Provides success message or error details

```python
# Create fresh connection for deletion operations
delete_conn = init_connection()
delete_cursor = delete_conn.cursor()

# Delete all related data - using individual statements instead of transaction
# 1. Delete quizzes (if table exists)
try:
    delete_cursor.execute("DELETE FROM quizzes WHERE student_id = %s", (student_id_to_delete,))
    delete_conn.commit()
except Exception as quiz_error:
    st.warning(f"Note: Could not delete quizzes: {str(quiz_error)}")

# 2. Delete tasks
delete_cursor.execute("DELETE FROM tasks WHERE student_id = %s", (student_id_to_delete,))
delete_conn.commit()

# 3. Delete notes
delete_cursor.execute("DELETE FROM notes WHERE student_id = %s", (student_id_to_delete,))
delete_conn.commit()

# 4. Delete knowledge_base entries related to the student
try:
    # Use string comparison instead of JSON operators if possible
    delete_cursor.execute("DELETE FROM knowledge_base WHERE metadata::text LIKE %s", 
                   (f'%"student_id": "{student_id_to_delete}"%',))
    delete_conn.commit()
except Exception as kb_error:
    st.warning(f"Note: Could not delete some knowledge base entries: {str(kb_error)}")

# 5. Finally delete the student
delete_cursor.execute("DELETE FROM students WHERE id = %s", (student_id_to_delete,))
delete_conn.commit()
```

## Error Handling

The Dashboard implements several error handling mechanisms:

1. **Table Existence Checks**: Uses try/except blocks when querying tables that might not exist
2. **Data Type Conversion**: Safely converts Decimal values to float to avoid type errors
3. **Empty Data Handling**: Checks for empty result sets before attempting visualization
4. **Connection Management**: Ensures database connections are properly closed, even after errors
5. **Transaction Management**: Uses individual commits to prevent transaction timeouts

## UI Components

### Tab Structure

```python
# Create tabs for different views
overview_tab, students_tab, activity_tab, performance_tab = st.tabs([
    "System Overview", "Student Leaderboard", "Activity Metrics", "Performance Analytics"
])
```

### Metric Display

```python
# Summary metrics in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Students", total_students)
with col2:
    st.metric("Total Notes", total_notes)
with col3:
    st.metric("Total Tasks", total_tasks)
with col4:
    st.metric("Total Quizzes", total_quizzes)
```

### Interactive Elements

```python
# Select a specific student to analyze
cursor.execute("SELECT id, name FROM students ORDER BY name")
students = cursor.fetchall()

if students:
    student_options = ["All Students"] + [f"{name} (ID: {id})" for id, name in students]
    selected_student = st.selectbox("Select Student:", student_options)
```

## Performance Considerations

1. **Query Optimization**: Uses COUNT(DISTINCT) and appropriate JOINs to minimize database load
2. **Data Limiting**: Restricts time-series queries to 30 days to manage data volume
3. **Visualization Efficiency**: Uses Streamlit's native charting when possible for better performance
4. **Connection Management**: Creates and closes connections appropriately to prevent resource leaks

## Known Limitations

1. The dashboard assumes the existence of certain database tables and columns
2. Large datasets may cause performance issues in visualization components
3. Student deletion process may encounter issues with complex foreign key relationships
4. Quiz performance analysis requires at least two quizzes for trend calculation

## Integration Points

The Dashboard integrates with other system components:

1. **Database Schema**: Relies on tables created by `init_db()` and `check_db_schema()`
2. **Session State**: Uses Streamlit's session state for user identification
3. **Connection Management**: Uses the shared `init_connection()` function

## Future Technical Enhancements

1. **Caching**: Implement Streamlit's caching for expensive database queries
2. **Pagination**: Add pagination for large datasets in leaderboards and tables
3. **Asynchronous Processing**: Use async operations for long-running deletion processes
4. **Data Export**: Add functionality to export analytics as CSV or Excel files
5. **Access Control**: Implement role-based access control for administrative functions 