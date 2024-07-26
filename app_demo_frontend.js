document.getElementById('search-form').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const name = document.getElementById('name').value;
    const location = document.getElementById('location').value;
    const age = document.getElementById('age').value;
    const income = document.getElementById('income').value;
    const coverage = document.getElementById('coverage').value;
    const healthHistory = document.getElementById('health-history').value;

    // Get insurance plans based on the user's criteria
    const plans = getInsurancePlans(age, income, coverage);
    
    // Display the plans along with user information
    displayPlans(plans, name, location, healthHistory);
});

function getInsurancePlans(age, income, coverage) {
    const plans = [
        { name: 'HealthPartners Gold', price: 250, coverage: 'individual', advantages: ['Low deductibles', 'Comprehensive coverage'], disadvantages: ['Higher premiums'], url: 'https://www.mnsure.org/healthpartners-gold', video: 'https://www.youtube.com/watch?v=example1' },
        { name: 'Medica Silver', price: 200, coverage: 'family', advantages: ['Moderate premiums'], disadvantages: ['Higher deductibles'], url: 'https://www.mnsure.org/medica-silver', video: 'https://www.youtube.com/watch?v=example2' },
        { name: 'Blue Cross Blue Shield Bronze', price: 150, coverage: 'individual', advantages: ['Lowest premiums'], disadvantages: ['High deductibles'], url: 'https://www.mnsure.org/blue-cross-blue-shield-bronze', video: 'https://www.youtube.com/watch?v=example3' }
    ];
    
    // Filter plans based on the user's coverage preference
    return plans.filter(plan => plan.coverage.toLowerCase() === coverage.toLowerCase());
}

function displayPlans(plans, name, location, healthHistory) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';
    
    // Display user information
    resultsDiv.innerHTML += `<p><strong>User Name:</strong> ${name}</p>`;
    resultsDiv.innerHTML += `<p><strong>Location:</strong> ${location}</p>`;
    resultsDiv.innerHTML += `<p><strong>Health History:</strong> ${healthHistory}</p>`;
    
    // Check if there are plans to display
    if (plans.length === 0) {
        resultsDiv.innerHTML += '<p>No plans available for the selected criteria.</p>';
    } else {
        plans.forEach(plan => {
            const planDiv = document.createElement('div');
            planDiv.classList.add('plan');
            planDiv.innerHTML = `
                <h3>${plan.name}</h3>
                <p>Price: $${plan.price} per month</p>
                <p>Coverage: ${plan.coverage}</p>
                <p><strong>Advantages:</strong> ${plan.advantages.join(', ')}</p>
                <p><strong>Disadvantages:</strong> ${plan.disadvantages.join(', ')}</p>
                <p><a href="${plan.url}" target="_blank">View Plan Details</a></p>
                <p><a href="${plan.video}" target="_blank">Watch Guide Video</a></p>
            `;
            resultsDiv.appendChild(planDiv);
        });
    }
}
