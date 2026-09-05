import random
import math

class DemocraticHammurabi:
    """
    Democratic Hammurabi with the Silver Shekel Monetary Economy.
    Decouples monetary capital (silver) from biological calories (grain).
    Rats eat grain, but royal silver vaults are 100% immune!
    """
    def __init__(self, max_years=12):
        self.max_years = max_years
        self.reset()

    def reset(self):
        self.year = 1
        self.population = 100
        self.grain = 3000            # Bushels of food & seed in silos
        self.land = 1000             # Acres of farmable land
        self.silver = 30000           # Silver shekels in royal vault (rats cannot eat silver!)
        self.elite_pop = int(self.population * 0.15)
        self.worker_pop = int(self.population * 0.5)
        self.farmer_pop = int(self.population * 0.35)
        self.land_demand = float(self.elite_pop * 40) + float(self.farmer_pop * 15) + float(self.worker_pop * 2)
        # Dynamic wealth baseline
        self.initial_pop = self.population
        self.initial_wealth = self.silver + (self.land * 25) + int(self.grain * 1.0)

        # Market prices
        self.land_price = random.randint(20, 30)                # Silver per acre
        self.grain_price = round(random.uniform(0.9, 1.2), 2)  # Silver per bushel

        # Faction approvals (0 to 100)
        self.farmers_approval = 50.0
        self.workers_approval = 50.0
        self.elites_approval = 50.0

        self.is_done = False
        self.game_over_reason = ""
        self.starved_total = 0
        self.last_rats_ate = 0
        self.last_harvest_yield = 3
        self.last_immigrants = 0

        return self._get_state()

    def _get_state(self):
        years_until_election = 4 - (self.year % 4)
        if years_until_election == 4:
            years_until_election = 0

        # 11-Dimensional State Vector
        return [
            float(self.year),
            float(self.population),
            float(self.grain),
            float(self.land),
            float(self.land_price),
            float(self.silver),
            float(self.grain_price),
            float(self.farmers_approval),
            float(self.workers_approval),
            float(self.elites_approval),
            float(years_until_election)
        ]

    def step(self, actions):
        """
        Actions is a list of 4 continuous values:
        - action_land: [-1, 1]. <0 means sell fraction of land for silver, >0 means buy land using silver.
        - action_grain_trade: [-1, 1]. <0 means export/sell grain for silver, >0 means import/buy grain using silver.
        - action_feed: [0, 1]. fraction of silo grain to use for feeding citizens.
        - action_plant: [0, 1]. fraction of remaining silo grain to plant as seed.
        """
        if self.is_done:
            return self._get_state(), 0, self.is_done, {"reason": "Already done"}

        action_land = float(actions[0])
        action_grain_trade = float(actions[1])
        action_feed = float(actions[2])
        action_plant = float(actions[3])

        # Clip actions to expected ranges
        action_land = max(-1.0, min(1.0, action_land))
        action_grain_trade = max(-1.0, min(1.0, action_grain_trade))
        action_feed = max(0.0, min(1.0, action_feed))
        action_plant = max(0.0, min(1.0, action_plant))

        # ----------------------------------------------------------------------
        # 1. Land Market (Traded in Silver Coins)
        # ----------------------------------------------------------------------
        land_changed = 0
        if action_land < 0:
            # Sell land for silver
            acres_to_sell = int(abs(action_land) * self.land)
            self.land -= acres_to_sell
            silver_earned = acres_to_sell * self.land_price
            self.silver += silver_earned
            land_changed = -acres_to_sell
        elif action_land > 0:
            # Buy land using silver
            max_acres_affordable = int(self.silver // self.land_price)
            acres_to_buy = int(action_land * max_acres_affordable)
            self.land += acres_to_buy
            self.silver -= acres_to_buy * self.land_price
            land_changed = acres_to_buy

        # ----------------------------------------------------------------------
        # 2. Grain Merchant Exchange (Buy/Sell Grain for Silver)
        # ----------------------------------------------------------------------
        grain_traded = 0
        if action_grain_trade < 0:
            # Export / Sell surplus grain for silver
            bushels_to_sell = int(abs(action_grain_trade) * self.grain)
            self.grain -= bushels_to_sell
            silver_earned = int(bushels_to_sell * self.grain_price)
            self.silver += silver_earned
            grain_traded = -bushels_to_sell
        elif action_grain_trade > 0:
            # Import / Buy emergency grain using silver
            max_grain_affordable = int(self.silver // self.grain_price)
            bushels_to_buy = int(action_grain_trade * max_grain_affordable)
            self.grain += bushels_to_buy
            self.silver -= int(bushels_to_buy * self.grain_price)
            grain_traded = bushels_to_buy

        # ----------------------------------------------------------------------
        # 3. Feed Citizens (from Silo Grain)
        # ----------------------------------------------------------------------
        grain_for_food = int(action_feed * self.grain)
        self.grain -= grain_for_food

        people_fed = grain_for_food // 20
        starved = max(0, self.population - people_fed)
        self.starved_total = starved

        if starved > 0:
            self.population -= starved

        # Immediate impeachment if starvation exceeds 45%
        if starved > 0.45 * (self.population + starved):
            self.is_done = True
            self.game_over_reason = "Impeached for extreme starvation"
            return self._get_state(), self._calculate_reward(), self.is_done, {"reason": self.game_over_reason}

        # ----------------------------------------------------------------------
        # 4. Planting Seeds (1 bushel per acre, max 10 acres per citizen)
        # ----------------------------------------------------------------------
        grain_for_planting = int(action_plant * self.grain)
        max_workable_by_people = self.population * 10
        actual_planted = min(grain_for_planting, self.land, max_workable_by_people)
        self.grain -= actual_planted

        # ----------------------------------------------------------------------
        # 5. Harvest & Rats
        # ----------------------------------------------------------------------
        yield_per_acre = random.randint(1, 5)
        self.last_harvest_yield = yield_per_acre
        harvest = actual_planted * yield_per_acre
        self.grain += harvest

        # Rats eat grain (Silver is 100% safe in vaults!)
        rats_ate = 0
        if random.random() < 0.4:  # 40% chance of rats
            rats_ate = int(self.grain * random.uniform(0.1, 0.3))
            self.grain -= rats_ate
        self.last_rats_ate = rats_ate

        # ----------------------------------------------------------------------
        # 6. Demographics (Births & Immigrants)
        # ----------------------------------------------------------------------
        immigrants = 0
        if starved == 0:
            # Prosperity brings merchants and settlers
            wealth_factor = (20 * self.land + self.grain + self.silver) / (100 * self.population + 1)
            immigrants = random.randint(1, 10) + int(wealth_factor)
            self.population += immigrants
            self.elite_pop = int(self.population * 0.15)
            self.farmer_pop = int(self.population * 0.35)
            self.worker_pop = self.population - self.elite_pop - self.farmer_pop

        self.last_immigrants = immigrants

        # ----------------------------------------------------------------------
        # 7. Faction Approval Updates
        # ----------------------------------------------------------------------
        # Farmers: Love holding/buying land (+5), hate selling land (-10), love 100% planting (+5)
        if land_changed > 0:
            self.farmers_approval += 5
        elif land_changed < 0:
            self.farmers_approval -= 10
        if actual_planted == self.land:
            self.farmers_approval += 5

        # Workers: Hate starvation, love low food prices and full bellies
        if starved > 0:
            self.workers_approval -= (starved / self.population) * 100
        else:
            self.workers_approval += 5

        # Elites: Value monetary wealth; expect wealth to keep pace with population growth (Per-Capita Wealth standard)
        total_wealth = self.silver + (self.land * self.land_price) + int(self.grain * self.grain_price)
        expected_wealth = int((self.initial_wealth / max(1, self.initial_pop)) * self.population)
        if total_wealth >= expected_wealth:
            self.elites_approval += 5
        else:
            self.elites_approval -= 5

        self.farmers_approval = self._clamp_and_decay_approval(self.farmers_approval)
        self.workers_approval = self._clamp_and_decay_approval(self.workers_approval)
        self.elites_approval = self._clamp_and_decay_approval(self.elites_approval)

        # ----------------------------------------------------------------------
        # 8. Democratic Elections (Every 4 years)
        # ----------------------------------------------------------------------
        if self.year % 4 == 0:
            total_voters = self.farmer_pop + self.worker_pop + self.elite_pop
            average_approval = ((self.farmer_pop * self.farmers_approval) + (self.worker_pop * self.workers_approval) + (self.elite_pop * self.elites_approval)) / total_voters
            if average_approval < 45.0:
                self.is_done = True
                self.game_over_reason = f"Lost election with {average_approval:.1f}% approval"
                return self._get_state(), self._calculate_reward(), self.is_done, {"reason": self.game_over_reason}

        # ----------------------------------------------------------------------
        # 9. Next Year Market Price Fluctuations
        # ----------------------------------------------------------------------
        self.year += 1
        self.land_price = random.randint(20, 30)

        # Grain price responds dynamically to harvest yield & scarcity
        if yield_per_acre <= 2 or rats_ate > 0:
            # Famine / Scarcity spikes grain prices!
            self.grain_price = round(random.uniform(1.3, 2.0), 2)
        elif yield_per_acre >= 4:
            # Bumper harvest lowers grain prices
            self.grain_price = round(random.uniform(0.7, 0.95), 2)
        else:
            # Normal harvest
            self.grain_price = round(random.uniform(0.95, 1.25), 2)

        if self.year > self.max_years:
            self.is_done = True
            self.game_over_reason = "Completed term successfully!"

        return self._get_state(), self._calculate_reward(), self.is_done, {
            "reason": self.game_over_reason,
            "harvest_yield": yield_per_acre,
            "rats_ate": rats_ate,
            "immigrants": immigrants,
        }

    def _clamp_and_decay_approval(self, approval):
        if approval > 50:
            approval -= (approval - 50) * 0.1
        elif approval < 50:
            approval += (50 - approval) * 0.1
        return max(0.0, min(100.0, approval))

    def _calculate_reward(self):
        survival_bonus = self.year * 100
        wealth_score = (self.silver + (self.land * self.land_price) + (self.grain * self.grain_price)) / 100.0
        pop_score = self.population * 2
        approval_score = min(self.farmers_approval, self.workers_approval, self.elites_approval) * 3.0
        penalty = self.starved_total * 50

        total_score = survival_bonus + wealth_score + pop_score + approval_score - penalty

        if self.is_done and self.year <= self.max_years:
            total_score = total_score / 10.0

        return total_score