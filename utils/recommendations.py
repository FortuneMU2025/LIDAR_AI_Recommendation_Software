def generate_recommendations(density_results, flow_results):
    """
    Generate actionable crowd management recommendations based on analysis results.
    
    Args:
        density_results (dict): Results from crowd density analysis
        flow_results (dict): Results from crowd flow analysis
        
    Returns:
        dict: Recommendations including issues, actions, and opportunities
    """
    recommendations = {
        'issues': [],
        'actions': [],
        'opportunities': []
    }
    
    # Analyze high density areas (hotspots)
    if density_results['hotspots']:
        for i, hotspot in enumerate(density_results['hotspots']):
            # Create issue for high density areas
            if hotspot['density'] > 3.0:  # Very high density
                issue = {
                    'title': f'Critical crowd density at location {i+1}',
                    'severity': min(10, int(hotspot['density'] * 2)),
                    'location': f'({hotspot["x"]:.1f}, {hotspot["y"]:.1f})',
                    'description': f'Extremely high crowd density of {hotspot["density"]:.2f} people/m² detected. This exceeds safety thresholds and creates significant safety risks.'
                }
                recommendations['issues'].append(issue)
                
                # Create corresponding action
                action = {
                    'title': f'Reduce density at hotspot {i+1}',
                    'priority': 'High',
                    'description': 'Immediate action required to reduce crowd density in this area to prevent potential safety incidents.',
                    'steps': [
                        'Deploy additional staff to redirect crowd flow away from this area',
                        'Consider temporarily restricting entry to this zone until density decreases',
                        'Use PA announcements to encourage people to move to less crowded areas',
                        'Open alternative pathways to reduce flow through this bottleneck'
                    ]
                }
                recommendations['actions'].append(action)
                
            elif hotspot['density'] > 2.0:  # High density
                issue = {
                    'title': f'High crowd density at location {i+1}',
                    'severity': min(8, int(hotspot['density'] * 2)),
                    'location': f'({hotspot["x"]:.1f}, {hotspot["y"]:.1f})',
                    'description': f'High crowd density of {hotspot["density"]:.2f} people/m² detected. This is approaching unsafe levels and requires attention.'
                }
                recommendations['issues'].append(issue)
                
                # Create corresponding action
                action = {
                    'title': f'Manage crowd at hotspot {i+1}',
                    'priority': 'Medium',
                    'description': 'Action required to prevent further density increase and maintain safe conditions.',
                    'steps': [
                        'Increase staff presence in this area to monitor crowd behavior',
                        'Create one-way flow systems to prevent counterflow and congestion',
                        'Consider timed entry or pulsed admission to this area',
                        'Provide clear signage directing to alternative routes'
                    ]
                }
                recommendations['actions'].append(action)
    
    # Analyze overall crowd density
    if density_results['avg_density'] > 2.5:
        issue = {
            'title': 'Overall crowd density too high',
            'severity': min(9, int(density_results['avg_density'] * 2)),
            'location': 'Entire venue',
            'description': f'The average crowd density of {density_results["avg_density"]:.2f} people/m² across the venue exceeds comfortable levels. This creates risk of overcrowding throughout the venue.'
        }
        recommendations['issues'].append(issue)
        
        action = {
            'title': 'Implement venue-wide density management',
            'priority': 'High',
            'description': 'Take immediate steps to reduce overall crowd density throughout the venue.',
            'steps': [
                'Temporarily restrict new entries until density decreases',
                'Open additional space if available',
                'Implement timed entry/exit systems',
                'Consider early closing of certain areas to gradually disperse crowds'
            ]
        }
        recommendations['actions'].append(action)
    
    # Analyze flow bottlenecks
    if flow_results['bottlenecks']:
        for i, bottleneck in enumerate(flow_results['bottlenecks']):
            if bottleneck['severity'] >= 7:  # Critical bottleneck
                issue = {
                    'title': f'Critical flow bottleneck at location {i+1}',
                    'severity': bottleneck['severity'],
                    'location': f'({bottleneck["x"]:.1f}, {bottleneck["y"]:.1f})',
                    'description': f'Severe crowd flow constriction detected with high risk of crowd compression and potential safety issues.'
                }
                recommendations['issues'].append(issue)
                
                action = {
                    'title': f'Resolve critical bottleneck {i+1}',
                    'priority': 'High',
                    'description': 'Immediate action required to resolve this flow bottleneck and prevent potential crowd crush incidents.',
                    'steps': [
                        'Deploy staff to actively manage crowd flow through this area',
                        'Implement one-way system to prevent counterflow',
                        'Consider widening the pathway if physically possible',
                        'Temporarily close this route and redirect traffic if alternative routes are available'
                    ]
                }
                recommendations['actions'].append(action)
                
            elif bottleneck['severity'] >= 4:  # Significant bottleneck
                issue = {
                    'title': f'Flow bottleneck at location {i+1}',
                    'severity': bottleneck['severity'],
                    'location': f'({bottleneck["x"]:.1f}, {bottleneck["y"]:.1f})',
                    'description': f'Crowd flow bottleneck detected that is causing congestion and reduced movement speed.'
                }
                recommendations['issues'].append(issue)
                
                action = {
                    'title': f'Improve flow at bottleneck {i+1}',
                    'priority': 'Medium',
                    'description': 'Action required to improve crowd flow through this area and prevent crowding.',
                    'steps': [
                        'Mark clear lanes with floor tape or portable barriers',
                        'Position staff to guide crowd movement',
                        'Use signage to indicate expected flow direction',
                        'Remove any temporary obstacles if present'
                    ]
                }
                recommendations['actions'].append(action)
    
    # Check for slow overall crowd flow
    if flow_results['avg_speed'] < 0.5:  # Slow average movement
        issue = {
            'title': 'Slow overall crowd movement',
            'severity': min(7, int((0.7 - flow_results['avg_speed']) * 10)),
            'location': 'Entire venue',
            'description': f'Average crowd movement speed of {flow_results["avg_speed"]:.2f} m/s is below optimal levels, indicating potential congestion throughout venue.'
        }
        recommendations['issues'].append(issue)
        
        action = {
            'title': 'Improve overall crowd flow',
            'priority': 'Medium',
            'description': 'Implement strategies to improve movement throughout the venue.',
            'steps': [
                'Review and optimize venue layout to reduce obstructions',
                'Implement clear one-way systems in high-traffic areas',
                'Consider staggered scheduling for different activities',
                'Use staff to identify and quickly resolve developing bottlenecks'
            ]
        }
        recommendations['actions'].append(action)
    
    # Generate optimization opportunities
    # 1. Underutilized spaces
    recommendations['opportunities'].append({
        'title': 'Identify and utilize low-density areas',
        'impact': 'Medium',
        'description': 'Areas with consistently low crowd density represent an opportunity to better distribute attendees and reduce pressure on high-density zones. Consider relocating popular attractions or services to these areas.'
    })
    
    # 2. Improved signage and information
    recommendations['opportunities'].append({
        'title': 'Dynamic information systems',
        'impact': 'High',
        'description': 'Implement real-time digital signage showing crowd density in different areas. This allows attendees to make informed decisions about which areas to visit, naturally balancing crowd distribution.'
    })
    
    # 3. Flow optimization
    recommendations['opportunities'].append({
        'title': 'Optimize crowd flow patterns',
        'impact': 'High',
        'description': 'The dominant crowd direction is ' + flow_results['dominant_direction'] + '. Design the venue layout to work with this natural flow direction rather than against it to reduce congestion and improve attendee experience.'
    })
    
    # 4. Entry/exit management
    recommendations['opportunities'].append({
        'title': 'Improved entry/exit management',
        'impact': 'Medium',
        'description': 'Consider implementing timed entry tickets or dynamic entry control based on real-time density data to prevent overcrowding from occurring in the first place.'
    })
    
    return recommendations
